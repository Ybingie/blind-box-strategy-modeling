from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class ModelConfig:
    roles: tuple[str, ...] = (
        'Light Yagami, front view',
        'Light Yagami, profile',
        'L, light-gray portrait',
        'L, dark-gray profile',
        'Misa Amane, smiling',
        'Misa Amane, choker',
        'Near',
        'Mello'
    )
    short_roles: tuple[str, ...] = (
        'Light-front',
        'Light-profile',
        'L-light gray',
        'L-dark gray',
        'Misa-smile',
        'Misa-choker',
        'Near',
        'Mello'
    )
    second_hand_prices: tuple[float, ...] = (60.0, 58.0, 60.0, 58.5, 37.75, 32.0, 20.33, 40.0)
    probabilities: tuple[float, ...] = (0.125,) * 8
    blind_price: float = 39.8
    resale_success_prob: float = 0.6
    platform_fee_rate: float = 0.006
    buyer_shipping: float = 6.0
    seller_cost: float = 4.0
    budget: float = 400.0
    fixed_k_max: int = 15
    base_seed: int = 20260917

    @property
    def n(self) -> int:
        return len(self.roles)

    @property
    def direct_cost(self) -> float:
        return float(sum(self.second_hand_prices))

    @property
    def resale_values(self) -> tuple[float, ...]:
        return tuple(
            (
                max(
                    (1 - self.platform_fee_rate) * (b - self.buyer_shipping) - self.seller_cost,
                    0.0
                ) for b in self.second_hand_prices
            )
        )

    @property
    def expected_resale_values(self) -> tuple[float, ...]:
        return tuple((self.resale_success_prob * g for g in self.resale_values))

@dataclass(frozen=True)
class RunConfig:
    simulations: int
    quick: bool
    output_root: Path
    png_dpi: int = 320

    @classmethod
    def build(cls, script_dir: Path, quick: bool) -> 'RunConfig':
        return cls(simulations=10000 if quick else 100000, quick=quick, output_root=script_dir / 'outputs')
