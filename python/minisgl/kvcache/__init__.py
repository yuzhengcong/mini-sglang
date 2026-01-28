from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

from minisgl.utils import Registry

if TYPE_CHECKING:
    import torch
    from minisgl.models import ModelConfig

from .base import (
    BaseCacheHandle,
    BaseCacheManager,
    BaseKVCache,
    KVCacheLayout,
    SizeInfo,
)


class CacheManagerCreator(Protocol):
    def __call__(self, device: torch.device) -> BaseCacheManager: ...


SUPPORTED_CACHE_MANAGER = Registry[CacheManagerCreator]("Cache Manager")


def create_kvcache(
    model_config: ModelConfig,
    num_pages: int,
    dtype: torch.dtype,
    device: torch.device,
    cache_layout: KVCacheLayout = KVCacheLayout.LayerFirst,
) -> BaseKVCache:
    from .mha_pool import MHAKVCache
    from .mla_pool import MLAKVCache
    
       

    latent_dim = getattr(model_config, "latent_dim", None)
    kv_variant = getattr(model_config, "kv_variant", None)
    use_mla = latent_dim is not None and kv_variant == "mla"

    if use_mla and MLAKVCache is not None:
        return MLAKVCache(
            num_kv_heads=model_config.num_kv_heads,
            num_pages=num_pages,
            kv_layout=cache_layout,
            num_layers=model_config.num_layers,
            latent_dim=int(latent_dim),  # type: ignore[arg-type]
            device=device,
            dtype=dtype,
        )
    else:
        return MHAKVCache(
            num_kv_heads=model_config.num_kv_heads,
            num_pages=num_pages,
            kv_layout=cache_layout,
            num_layers=model_config.num_layers,
            head_dim=model_config.head_dim,
            device=device,
            dtype=dtype,
        )


@SUPPORTED_CACHE_MANAGER.register("naive")
def create_naive_cache_manager(device: torch.device):
    from .naive_manager import NaiveCacheManager

    return NaiveCacheManager(device=device)


@SUPPORTED_CACHE_MANAGER.register("radix")
def create_radix_cache_manager(device: torch.device):
    from .radix_manager import RadixCacheManager

    return RadixCacheManager(device=device)


def create_cache_manager(device: torch.device, type: str) -> BaseCacheManager:
    return SUPPORTED_CACHE_MANAGER[type](device)


__all__ = [
    "create_kvcache",
    "create_cache_manager",
    "BaseKVCache",
    "KVCacheLayout",
    "BaseCacheHandle",
    "BaseCacheManager",
    "SizeInfo",
    "SUPPORTED_CACHE_MANAGER",
]
