"""Punto de extensión para la etapa de comparables (desapalancamiento /
reapalancamiento de Beta, método Hamada). NO implementado todavía a propósito:
esta primera versión solo deja funcionando el Beta directo de la empresa.

Cuando corresponda avanzar a esta etapa, el flujo será:

1. Beta de empresas comparables (mismo procedimiento que `regression.py`,
   una corrida por comparable).
2. Estructura de capital de cada comparable (B/P).
3. Desapalancar cada Beta observado:
       beta_U = beta_L / (1 + (1 - t_c) * (B/P))
4. Promediar los Beta desapalancados del negocio.
5. Definir la estructura de capital objetivo de la empresa en estudio.
6. Reapalancar con esa estructura objetivo:
       beta_L_obj = beta_U * (1 + (1 - t_c) * (B/P)_obj)

Las firmas de abajo son el contrato para cuando se implemente; hoy solo
lanzan `NotImplementedError` a propósito, para que quede explícito que no
hay que llamarlas.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ComparableInput:
    name: str
    beta_levered: float
    debt_to_equity: float  # B/P
    tax_rate: float


def unlever_beta(comparable: ComparableInput) -> float:
    """beta_U = beta_L / (1 + (1 - t_c) * (B/P))  — fórmula de Hamada."""
    raise NotImplementedError("Etapa de comparables: pendiente, ver sección 11 del encargo.")


def relever_beta(beta_unlevered: float, target_debt_to_equity: float, tax_rate: float) -> float:
    """beta_L_obj = beta_U * (1 + (1 - t_c) * (B/P)_obj)"""
    raise NotImplementedError("Etapa de comparables: pendiente, ver sección 11 del encargo.")
