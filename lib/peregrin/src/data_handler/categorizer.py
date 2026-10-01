from __future__ import annotations

import polars as pl
from typing import Any

from ..settings import params
from .._pckg_exceptions._pckg_errors import *
from .._pckg_exceptions._pckg_warnings import *



class Categorizer:

    DEFAULT_CATEGORIES = ['set', 'subset', 'group', 'subgroup', 'subsubgroup']


    def __init__(self): ...


    def categorize(
        self,
        data: pl.DataFrame,
        keep_which: dict[str, list] | None = None,
        *,
        aggby: list | None = None,
        aggdict: dict[str, Any] | list[pl.Expr] | None = None,
        **kwargs
    ) -> pl.DataFrame:
        """
        Categorize and aggregate data.

        Parameters
        ----------
        data : pl.DataFrame
            The input DataFrame to be categorized and aggregated

        keep_which : dict[str, list], optional
            A dictionary containing the keys = columns with given keys that are to be retained `{<column>: list[<selected values>]}`.

        aggby : list, optional
            A list of columns to group by for aggregation. Default is an empty list.

        aggdict : dict | list[pl.Expr], optional
            Aggregations to apply. Either a dict `{<column>: <func name> | list[<func names>]}`
            (e.g. `{'speed': 'mean', 'dist': ['sum', 'max']}`) or a list of Polars expressions.
        """

        self.data = data
        self.keep_which = keep_which if keep_which is not None else {}
        self.aggby = aggby if aggby is not None else []
        self.aggdict = aggdict if aggdict is not None else {}

        self._checkcats()
        self._filter()

        if self.aggdict and self.aggby:
            self._aggregate()

        return self.data


    def _checkcats(self) -> None:
        """ Check for errors in the provided categories and values. """

        for cat, vals in self.keep_which.items():
            if cat not in self.data.columns:
                raise CategorizerError(f"Column '{cat}' not found in DataFrame.")

            present = set(self.data.get_column(cat).unique().to_list())
            for val in vals:
                if val not in present:
                    raise CategorizerError(f"Value '{val}' not found in column '{cat}'.")


    def _filter(self) -> None:
        """ Filter DataFrame categories. """

        for cat, vals in self.keep_which.items():
            try:
                self.data = self.data.filter(pl.col(cat).is_in(vals))
            except Exception as e:
                raise CategorizerError(f"Error filtering data category: '{cat}': {e}")


    def _build_agg_exprs(self) -> list[pl.Expr]:
        """ Translate a pandas-style aggregation dict into Polars expressions. """

        if isinstance(self.aggdict, list):
            return self.aggdict

        exprs: list[pl.Expr] = []
        for col, funcs in self.aggdict.items():
            if isinstance(funcs, pl.Expr):
                exprs.append(funcs.alias(col))
                continue

            multiple = isinstance(funcs, (list, tuple))
            for func in (funcs if multiple else [funcs]):
                if not isinstance(func, str):
                    raise CategorizerError(
                        f"Unsupported aggregation for column '{col}': {func!r}. Use a string name or pl.Expr."
                    )
                name = {'nunique': 'n_unique', 'size': 'len'}.get(func, func)
                try:
                    expr = getattr(pl.col(col), name)()
                except AttributeError:
                    raise CategorizerError(f"Unknown aggregation '{func}' for column '{col}'.")
                exprs.append(expr.alias(f"{col}_{func}" if multiple else col))
        return exprs


    def _aggregate(self) -> None:
        """ Aggregate the filtered DataFrame. """

        try:
            self.data = (
                self.data
                .group_by(self.aggby, maintain_order=True)
                .agg(self._build_agg_exprs())
                .sort(self.aggby)
            )
        except CategorizerError:
            raise
        except Exception as e:
            raise CategorizerError(f"Error aggregating data: {e}")



categorize = Categorizer().categorize