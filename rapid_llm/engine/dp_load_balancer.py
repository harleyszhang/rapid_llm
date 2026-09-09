"""Load-balancing policies for the data-parallel router.

Every policy is a :class:`LoadBalancer` asked where a new request should
go, so the router code never branches on policy specifics;
``make_load_balancer`` builds one by registered name.

Usage:
    balancer = make_load_balancer("round_robin", dp_size)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections import OrderedDict, deque
from collections.abc import Sequence

from .prefix_cache import PREFIX_CACHE_BLOCK_SIZE, iter_block_hashes

#: Policy names accepted by :func:`make_load_balancer`, spelled as SGLang's
#: ``LoadBalanceMethod`` and its router spell them so that a reader who knows one
#: knows the other.
LOAD_BALANCERS = ("round_robin", "total_requests", "total_tokens", "cache_aware")


class LoadBalancer(ABC):
    """Chooses a replica per request, and is told when a request frees one.

    Args:
        dp_size: Number of replicas to choose between.

    Raises:
        ValueError: If ``dp_size`` is below 1.
    """

    #: Whether :meth:`select` and :meth:`release` read ``estimated_tokens``. The
    #: router skips tokenising the batch when every policy in play says ``False``.
    needs_token_estimate: bool = False

    #: Whether :meth:`select` reads ``token_ids``. Costs the same tokenizer pass as
    #: :attr:`needs_token_estimate` and subsumes it -- a policy holding the ids can
    #: count them itself -- so a cache-aware policy declares this one alone.
    needs_token_ids: bool = False

    def __init__(self, dp_size: int) -> None:
        if dp_size < 1:
            raise ValueError(f"dp_size must be >= 1, got {dp_size}")
        self.dp_size = dp_size

    @abstractmethod
    def select(self, estimated_tokens: int = 0, token_ids: Sequence[int] | None = None) -> int:
        """Return the replica index this request should go to.

        Args:
            estimated_tokens: Prompt length in tokens. Read only by policies whose
                :attr:`needs_token_estimate` is ``True``; passing it unconditionally
                keeps the call sites uniform.
            token_ids: The prompt's tokens. Read only by policies whose
                :attr:`needs_token_ids` is ``True``, and ``None`` otherwise for the
                same reason.
        """

    def release(  # noqa: B027
        self, replica: int, estimated_tokens: int = 0
    ) -> None:
        """Note that a request on ``replica`` has finished.

        The default does nothing on purpose: load-unaware policies keep no counts, so
        this is an optional hook rather than an abstract method. Only load-aware
        policies override it. Kept on the base class so the router calls it
        unconditionally.

        Args:
            replica: The replica the request ran on.
            estimated_tokens: The same estimate that was passed to :meth:`select`, so
                a token-aware policy can subtract exactly what it added.
        """


class RoundRobinBalancer(LoadBalancer):
    """Hand replicas out in turn, ignoring how long each request runs.

    The cheapest policy and the right default for offline batches, where the prompts
    are known together and no single one dominates. It is exactly SGLang's
    ``round_robin_scheduler`` minus the liveness skip (an offline pool has no replicas
    dropping out mid-run). Striping the requests rather than handing each replica a
    contiguous slice mixes long and short prompts evenly, so no one replica inherits
    every long prompt from an already-sorted list.
    """

    def __init__(self, dp_size: int) -> None:
        super().__init__(dp_size)
        self._counter = 0

    def select(self, estimated_tokens: int = 0, token_ids: Sequence[int] | None = None) -> int:
        replica = self._counter % self.dp_size
        self._counter += 1
        return replica


class _CountingBalancer(LoadBalancer):
    """Send each request to the replica with the smallest running total.

    The two load-aware policies differ only in *what* they count, so the arithmetic —
    pick the minimum, add on select, subtract on release — is written once and the
    subclass supplies the weight of one request via :meth:`weigh`. Lowest index breaks
    ties, so an idle pool degenerates to round-robin.
    """

    def __init__(self, dp_size: int) -> None:
        super().__init__(dp_size)
        self._load = [0] * dp_size

    @abstractmethod
    def weigh(self, estimated_tokens: int) -> int:
        """How much this request adds to a replica's running total."""

    def select(self, estimated_tokens: int = 0, token_ids: Sequence[int] | None = None) -> int:
        replica = min(range(self.dp_size), key=lambda i: self._load[i])
        self._load[replica] += self.weigh(estimated_tokens)
        return replica

    def release(self, replica: int, estimated_tokens: int = 0) -> None:
        self._load[replica] = max(0, self._load[replica] - self.weigh(estimated_tokens))

    @property
    def load(self) -> tuple[int, ...]:
        """Current running total per replica — the state a visualiser plots."""
        return tuple(self._load)


class TotalRequestsBalancer(_CountingBalancer):
    """Fewest in-flight *requests* wins — SGLang's ``total_requests``.

    Every request weighs 1, so this is the right policy when prompts are roughly
    interchangeable and what matters is not stacking four requests on one replica while
    another idles.
    """

    def weigh(self, estimated_tokens: int) -> int:
        return 1


class TotalTokensBalancer(_CountingBalancer):
    """Fewest in-flight *tokens* wins — SGLang's ``total_tokens``.

    Prefill cost is linear in prompt length, so with skewed lengths request counts are
    the wrong unit: two 4k prompts are not the load of two 40-token ones. It is the only
    policy that reads ``estimated_tokens``, and it declares so — ``cache_aware`` costs
    the router the same tokenizer pass but asks for the ids instead.

    A request weighs at least 1 so that a batch of empty prompts still round-robins
    instead of piling onto replica 0.
    """

    needs_token_estimate = True

    def weigh(self, estimated_tokens: int) -> int:
        return max(1, estimated_tokens)


class CacheAwareBalancer(LoadBalancer):
    """Fewest tokens *left to prefill* wins — SGLang's router's ``cache_aware``.

    Every replica keeps its own :class:`~rapid_llm.engine.prefix_cache.PrefixCache`,
    so a load-only policy scatters requests that share a prefix and makes ``dp_size``
    replicas prefill the same system prompt ``dp_size`` times. Affinity alone is not the
    fix either: whoever owns the popular prefix then attracts every request carrying it.

    Both concerns are one quantity measured once — how many tokens of this prompt would
    actually have to be prefilled on replica ``r``::

        cost(r) = outstanding_tokens(r) + max(1, prompt_tokens - cached_tokens(r))

    and the pick is the argmin. A long cached prefix makes a replica cheap; already
    being busy makes it expensive again. No weighting constant has to be invented to
    trade affinity against load, because both terms are tokens of prefill.

    With an empty index every ``cached_tokens`` is zero and the formula *is*
    :class:`TotalTokensBalancer`. That is what makes this a safe default rather than a
    gamble: it can only diverge from load balancing where it has a reason to.

    The index is the router's own approximation of what the replicas hold, built from
    what it routed rather than from anything they report. It is therefore wrong in both
    directions, and the two errors do not cost the same. A false positive — the replica
    evicted that prefix already — routes a request that then merely misses, costing
    balance. A false negative loses a reuse that was there. Neither can produce a wrong
    token, because the replica consults its real cache regardless, and that asymmetry is
    what makes an approximation the right tool instead of a reporting channel.

    Args:
        dp_size: Number of replicas.
        block_size: Tokens per hashed block. Must equal the replicas'
            :data:`~rapid_llm.engine.prefix_cache.PREFIX_CACHE_BLOCK_SIZE`, or the
            router's keys name nothing the replicas ever stored.
        hash_seed: Must equal the replicas' ``PrefixCache`` seed, for the same reason.
        index_capacity: Resident blocks remembered per replica. Bounds the router the
            way ``Scheduler._default_prefix_capacity`` bounds a replica: unbounded, a
            long-lived router accumulates one entry per block it has ever seen. Biased
            high on purpose, over-estimating residency being the cheaper error.

    Raises:
        ValueError: On a non-positive ``dp_size``, ``block_size`` or ``index_capacity``.
    """

    needs_token_ids = True

    #: Blocks per replica the router will remember. At the default block size that is
    #: a megatoken of prefix per replica, well above any single replica's KV cache.
    DEFAULT_INDEX_CAPACITY = 65536

    def __init__(
        self,
        dp_size: int,
        *,
        block_size: int = PREFIX_CACHE_BLOCK_SIZE,
        hash_seed: int = 0,
        index_capacity: int = DEFAULT_INDEX_CAPACITY,
    ) -> None:
        super().__init__(dp_size)
        if block_size < 1:
            raise ValueError(f"block_size must be >= 1, got {block_size}")
        if index_capacity < 1:
            raise ValueError(f"index_capacity must be >= 1, got {index_capacity}")
        self.block_size = block_size
        self.hash_seed = hash_seed
        self.index_capacity = index_capacity
        #: Per replica, the block hashes believed resident, least- to most-recently
        #: used. An ``OrderedDict`` used as a set, mirroring ``PrefixCache._blocks``.
        self._resident: list[OrderedDict[int, None]] = [OrderedDict() for _ in range(dp_size)]
        self._load = [0] * dp_size
        #: Charges still outstanding per replica, so :meth:`release` can subtract
        #: exactly what :meth:`select` added without being told which request ended.
        self._charges: list[deque[int]] = [deque() for _ in range(dp_size)]

    # ------------------------------------------------------------------ policy #
    def select(self, estimated_tokens: int = 0, token_ids: Sequence[int] | None = None) -> int:
        hashes = (
            list(iter_block_hashes(token_ids, self.block_size, self.hash_seed)) if token_ids else []
        )
        # ``token_ids`` is the honest length; ``estimated_tokens`` is the fallback for
        # a caller that only has the count, which degrades this to ``total_tokens``.
        prompt_tokens = len(token_ids) if token_ids else max(0, estimated_tokens)

        best_replica, best_cost, best_cached = 0, None, 0
        for replica in range(self.dp_size):
            cached = self._cached_blocks(replica, hashes)
            cost = self._load[replica] + self._uncached(prompt_tokens, cached)
            if best_cost is None or cost < best_cost:
                best_replica, best_cost, best_cached = replica, cost, cached

        charge = self._uncached(prompt_tokens, best_cached)
        self._load[best_replica] += charge
        self._charges[best_replica].append(charge)
        self._remember(best_replica, hashes)
        return best_replica

    def release(self, replica: int, estimated_tokens: int = 0) -> None:
        """Retire one charge on ``replica``; the prefix it cached stays indexed.

        Charges are popped oldest-first rather than matched to the request that ended,
        because this interface carries no request id. The aggregate is exact either way
        -- load stays the sum of what is still outstanding -- and the aggregate is all
        the load term reads. A release for a replica never selected is ignored, so a
        stray one cannot drive the load negative.
        """
        charges = self._charges[replica]
        if charges:
            self._load[replica] -= charges.popleft()

    # ------------------------------------------------------------------- index #
    def _uncached(self, prompt_tokens: int, cached_blocks: int) -> int:
        """What prefilling this prompt still costs on a replica holding those blocks.

        Floored at 1 so a fully cached prompt still weighs something: otherwise a
        popular prefix would be free, and free work piles up without limit.
        """
        return max(1, prompt_tokens - cached_blocks * self.block_size)

    def _cached_blocks(self, replica: int, hashes: Sequence[int]) -> int:
        """Leading blocks of ``hashes`` the router believes ``replica`` holds.

        The walk stops at the first miss instead of counting matches further along:
        the hashes are chained, so a later block is reusable only if every block
        before it is too.
        """
        resident = self._resident[replica]
        cached = 0
        for block_hash in hashes:
            if block_hash not in resident:
                break
            cached += 1
        return cached

    def _remember(self, replica: int, hashes: Sequence[int]) -> None:
        """Record that ``replica`` will hold these blocks, evicting LRU past capacity."""
        resident = self._resident[replica]
        for block_hash in hashes:
            resident[block_hash] = None
            resident.move_to_end(block_hash)
        while len(resident) > self.index_capacity:
            resident.popitem(last=False)

    # ----------------------------------------------------------- observability #
    @property
    def load(self) -> tuple[int, ...]:
        """Outstanding uncached-prefill tokens per replica — what :meth:`select` ranks."""
        return tuple(self._load)

    def cached_tokens(self, token_ids: Sequence[int], replica: int) -> int:
        """Leading tokens of ``token_ids`` the router believes ``replica`` has cached."""
        hashes = list(iter_block_hashes(token_ids, self.block_size, self.hash_seed))
        return self._cached_blocks(replica, hashes) * self.block_size

    def resident_blocks(self, replica: int) -> int:
        """How many blocks the router currently indexes for ``replica``."""
        return len(self._resident[replica])


def make_load_balancer(policy: str, dp_size: int) -> LoadBalancer:
    """Build the balancer named by ``policy``.

    Args:
        policy: One of :data:`LOAD_BALANCERS`.
        dp_size: Number of replicas.

    Raises:
        ValueError: On an unknown policy name.
    """
    builders = {
        "round_robin": RoundRobinBalancer,
        "total_requests": TotalRequestsBalancer,
        "total_tokens": TotalTokensBalancer,
        "cache_aware": CacheAwareBalancer,
    }
    if policy not in builders:
        raise ValueError(f"unknown load-balancer {policy!r}; choose from {LOAD_BALANCERS}")
    return builders[policy](dp_size)
