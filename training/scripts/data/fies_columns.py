"""
FIES Column Mapping Module

Maps FIES variable IDs to CSV column names and provides safe column access.
Based on Odin-Paper/Model/2_Data Collection/FIES Dictionary & Valueset.csv

Usage:
    from fies_columns import FIESColumns, get_column, validate_columns
    
    fies = FIESColumns(df)
    income = fies.get("TOINC")
    region = fies.get("W_REGN")
"""

from dataclasses import dataclass
from typing import Optional

import pandas as pd


# FIES variable ID -> CSV column name mapping
# Source: FIES Dictionary & Valueset.csv
FIES_COLUMN_MAP = {
    # IDs
    "W_REGN": "Region",
    "W_PROV": "Province",
    "SEQ_NO": "SEQ_NO",
    
    # Household Summary
    "RPROV": "Province Recode",
    "FSIZE": "Total Number of Family members",
    "PERCAPITA": "Per Capita Income",
    
    # Income Variables
    "REG_SAL": "REG_SAL",
    "SEASON_SAL": "SEASON_SAL",
    "WAGES": "Wages",
    "NETSHARE": "NETSHARE",
    "CASH_ABROAD": "CASH_ABROAD",
    "CASH_DOMESTIC": "CASH_DOMESTIC",
    "RENTALS_REC": "RENTALS_REC",
    "INTEREST": "INTEREST",
    "PENSION": "PENSION",
    "DIVIDENDS": "DIVIDENDS",
    "OTHER_SOURCE": "OTHER_SOURCE",
    "NET_RECEIPT": "NET_RECEIPT",
    "REGFT": "REGFT",
    "EAINC": "Total Income from Entrepreneurial Acitivites",
    "LOSSES": "LOSSES",
    "TOINC": "Total Household Income",
    
    # Food Expenditure
    "BREAD": "Bread and Cereals Expenditure",
    "MEAT": "Meat Expenditure",
    "FISH": "Total Fish and  marine products Expenditure",
    "MILK": "MILK",
    "OIL": "OIL",
    "FRUIT": "Fruit Expenditure",
    "VEG": "Vegetables Expenditure",
    "SUGAR": "SUGAR",
    "FOOD_NEC": "FOOD_NEC",
    "FRUIT_VEG": "FRUIT_VEG",
    "COFFEE": "COFFEE",
    "TEA": "TEA",
    "COCOA": "COCOA",
    "WATER": "WATER",
    "SOFTDRINKS": "SOFTDRINKS",
    "OTHER_NON_ALCOHOL": "OTHER_NON_ALCOHOL",
    "ALCOHOL": "Alcoholic Beverages Expenditure",
    "TOBACCO": "Tobacco Expenditure",
    "OTHER_VEG": "OTHER_VEG",
    "SERVICES_PRIMARY_GOODS": "SERVICES_PRIMARY_GOODS",
    "ALCOHOL_PROCDUCTION_SERVICES": "ALCOHOL_PROCDUCTION_SERVICES",
    "FOOD_HOME": "FOOD_HOME",
    "FOOD_OUTSIDE": "FOOD_OUTSIDE",
    "FOOD": "Total Food Expenditure",
    
    # Non-Food Expenditure
    "CLOTH": "Clothing, Footwear and Other Wear Expenditure",
    "HOUSING_WATER": "Housing and water Expenditure",
    "ACTRENT": "ACTRENT",
    "IMPUTED_RENT": "Imputed House Rental Value",
    "BIMPUTED_RENT": "BIMPUTED_RENT",
    "RENTVAL": "RENTVAL",
    "FURNISHING": "FURNISHING",
    "HEALTH": "Medical Care Expenditure",
    "TRANSPORT": "Transportation Expenditure",
    "COMMUNICATION": "Communication Expenditure",
    "RECREATION": "RECREATION",
    "EDUCATION": "Education Expenditure",
    "INSURANCE": "INSURANCE",
    "MISCELLANEOUS": "Miscellaneous Goods and Services Expenditure",
    "DURABLE": "DURABLE",
    "OCCASION": "Special Occasions Expenditure",
    "OTHER_EXPENDITURE": "OTHER_EXPENDITURE",
    "OTHER_DISBURSEMENT": "OTHER_DISBURSEMENT",
    "FOOD_ACCOM_SRVC": "FOOD_ACCOM_SRVC",
    "NFOOD": "NFOOD",
    
    # Totals
    "TOTEX": "TOTEX",
    "TOTDIS": "TOTDIS",
    "OTHREC": "OTHREC",
    "TOREC": "TOREC",
    
    # Household Characteristics
    "HEAD_SEX": "Household Head Sex",
    "HEAD_AGE": "Household Head Age",
    "HEAD_MARITAL": "Household Head Marital Status",
    "HEAD_EDUC": "Household Head Highest Grade Completed",
    "HEAD_JOB": "Household Head Job or Business Indicator",
    "HEAD_OCCUP": "Household Head Occupation",
    "HEAD_CLASS": "Household Head Class of Worker",
    "HH_TYPE": "Type of Household",
    "HH_SIZE": "Total Number of Family members",
    "MEMBERS_0_4": "Members with age less than 5 year old",
    "MEMBERS_5_17": "Members with age 5 - 17 years old",
    "MEMBERS_EMPLOYED": "Total number of family members employed",
    
    # Housing Characteristics
    "BUILDING": "Type of Building/House",
    "ROOF": "Type of Roof",
    "WALLS": "Type of Walls",
    "FLOOR_AREA": "House Floor Area",
    "HOUSE_AGE": "House Age",
    "BEDROOMS": "Number of bedrooms",
    "TENURE": "Tenure Status",
    "TOILET": "Toilet Facilities",
    "ELECTRICITY": "Electricity",
    "WATER_SOURCE": "Main Source of Water Supply",
    
    # Durable Goods
    "TV": "Number of Television",
    "CD_DVD": "Number of CD/VCD/DVD",
    "STEREO": "Number of Component/Stereo set",
    "FRIDGE": "Number of Refrigerator/Freezer",
    "WASHER": "Number of Washing Machine",
    "AC": "Number of Airconditioner",
    "CAR": "Number of Car, Jeep, Van",
    "PHONE_LANDLINE": "Number of Landline/wireless telephones",
    "PHONE_CELL": "Number of Cellular phone",
    "COMPUTER": "Number of Personal Computer",
    "STOVE": "Number of Stove with Oven/Gas Range",
    "BANCA": "Number of Motorized Banca",
    "MOTORCYCLE": "Number of Motorcycle/Tricycle",
    
    # Weights and PSU
    "RPSU": "RPSU",
    "RFACT": "RFACT",
    "MEM_RFACT": "MEM_RFACT",
    
    # Urban/Rural and Income Deciles
    "URB": "URB",
    "NPCINC": "NPCINC",
    "RPCINC": "RPCINC",
    "PRPCINC": "PRPCINC",
    "PPCINC": "PPCINC",
    "RPCINC_NIR": "RPCINC_NIR",
    "W_REGN_NIR": "W_REGN_NIR",
}

# Reverse mapping: CSV column name -> FIES variable ID
REVERSE_COLUMN_MAP = {v: k for k, v in FIES_COLUMN_MAP.items()}

# Critical columns that must exist for the pipeline
CRITICAL_COLUMNS = {
    "income": ["TOINC"],
    "food": ["FOOD"],
    "housing": ["HOUSING_WATER"],
    "transport": ["TRANSPORT"],
    "health": ["HEALTH"],
    "education": ["EDUCATION"],
    "region": ["W_REGN"],
    "household_size": ["HH_SIZE", "FSIZE"],
}

# Optional columns (used if available)
OPTIONAL_COLUMNS = {
    "head_age": ["HEAD_AGE"],
    "head_sex": ["HEAD_SEX"],
    "head_education": ["HEAD_EDUC"],
    "electricity": ["ELECTRICITY"],
    "water_source": ["WATER_SOURCE"],
    "tenure": ["TENURE"],
}


class FIESColumnError(Exception):
    """Raised when a required FIES column is missing."""
    pass


class FIESColumnWarning(UserWarning):
    """Warning for optional missing FIES columns."""
    pass


@dataclass
class ColumnInfo:
    """Information about a resolved column."""
    fies_id: str
    csv_name: str
    found: bool
    alternative: Optional[str] = None


class FIESColumns:
    """Safe column accessor for FIES DataFrames."""
    
    def __init__(self, df: pd.DataFrame, strict: bool = False):
        """
        Initialize FIES column accessor.
        
        Args:
            df: pandas DataFrame with FIES data
            strict: If True, raise error on missing critical columns
        
        Raises:
            FIESColumnError: If strict=True and critical columns are missing
        """
        self.df = df
        self.strict = strict
        self.resolved_columns: dict[str, ColumnInfo] = {}
        self._resolve_columns()
    
    def _resolve_columns(self):
        """Resolve all column mappings, finding alternatives if needed."""
        for fies_id, csv_name in FIES_COLUMN_MAP.items():
            if csv_name in self.df.columns:
                # Exact match on mapped CSV column name
                self.resolved_columns[fies_id] = ColumnInfo(
                    fies_id=fies_id,
                    csv_name=csv_name,
                    found=True,
                )
            elif fies_id in self.df.columns:
                # CSV uses FIES variable ID as column header directly
                self.resolved_columns[fies_id] = ColumnInfo(
                    fies_id=fies_id,
                    csv_name=fies_id,
                    found=True,
                )
            else:
                # Try to find alternative columns
                alternative = self._find_alternative_column(fies_id, csv_name)
                if alternative:
                    self.resolved_columns[fies_id] = ColumnInfo(
                        fies_id=fies_id,
                        csv_name=csv_name,
                        found=True,
                        alternative=alternative,
                    )
                else:
                    self.resolved_columns[fies_id] = ColumnInfo(
                        fies_id=fies_id,
                        csv_name=csv_name,
                        found=False,
                    )
    
    def _find_alternative_column(self, fies_id: str, csv_name: str) -> Optional[str]:
        """Find alternative column name if exact match not found."""
        # Common alternative names
        alternatives = {
            "TOINC": ["Total Income", "income", "HHIncome", "total_income"],
            "FOOD": ["Food Expenditure", "food_expense", "food"],
            "HOUSING_WATER": ["Housing", "housing", "housing_expense", "rent"],
            "TRANSPORT": ["Transportation", "transport", "transport_expense"],
            "HEALTH": ["Health", "health", "health_expense", "medical"],
            "EDUCATION": ["Education", "education", "education_expense", "school"],
            "HH_SIZE": ["Family Size", "household_size", "FSIZE", "size"],
            "W_REGN": ["Region", "region", "REGN"],
            "HEAD_AGE": ["Age", "age", "head_age"],
            "HEAD_SEX": ["Sex", "sex", "gender", "head_gender"],
        }
        
        if fies_id in alternatives:
            for alt in alternatives[fies_id]:
                if alt in self.df.columns:
                    return alt
        
        return None
    
    def get(self, fies_id: str, default=None) -> pd.Series:
        """
        Get a column by FIES variable ID.
        
        Args:
            fies_id: FIES variable ID (e.g., "TOINC", "FOOD")
            default: Value to return if column not found
        
        Returns:
            pandas Series with column data
        
        Raises:
            FIESColumnError: If column not found and no default provided
        """
        if fies_id not in self.resolved_columns:
            raise FIESColumnError(
                f"Unknown FIES variable ID: {fies_id}. "
                f"Valid IDs: {list(FIES_COLUMN_MAP.keys())}"
            )
        
        info = self.resolved_columns[fies_id]
        
        if not info.found:
            if default is not None:
                return pd.Series([default] * len(self.df), index=self.df.index)
            
            if self.strict:
                raise FIESColumnError(
                    f"Required column not found: {fies_id} (expected '{info.csv_name}')"
                )
            
            return None
        
        # Use alternative column if primary not found
        col_name = info.alternative if info.alternative else info.csv_name
        return self.df[col_name]
    
    def safe_get(self, fies_id: str, default=None) -> Optional[pd.Series]:
        """
        Safely get a column, returning default if not found.
        
        Args:
            fies_id: FIES variable ID
            default: Value to return if column not found
        
        Returns:
            pandas Series or default value
        """
        try:
            return self.get(fies_id, default=default)
        except FIESColumnError:
            return default
    
    def validate_critical_columns(self) -> list[str]:
        """
        Validate that all critical columns exist.
        
        Returns:
            List of missing critical column IDs
        """
        missing = []
        for category, column_ids in CRITICAL_COLUMNS.items():
            for col_id in column_ids:
                if col_id in self.resolved_columns and not self.resolved_columns[col_id].found:
                    missing.append(col_id)
                elif col_id not in self.resolved_columns:
                    missing.append(col_id)
        return missing
    
    def validate_optional_columns(self) -> dict[str, list[str]]:
        """
        Validate optional columns.
        
        Returns:
            Dict of category -> list of missing column IDs
        """
        missing = {}
        for category, column_ids in OPTIONAL_COLUMNS.items():
            category_missing = []
            for col_id in column_ids:
                if col_id in self.resolved_columns and not self.resolved_columns[col_id].found:
                    category_missing.append(col_id)
                elif col_id not in self.resolved_columns:
                    category_missing.append(col_id)
            if category_missing:
                missing[category] = category_missing
        return missing
    
    def get_column_report(self) -> dict:
        """Get a report of all resolved columns."""
        found = []
        missing = []
        alternatives_used = []
        
        for fies_id, info in self.resolved_columns.items():
            if info.found:
                found.append(fies_id)
                if info.alternative:
                    alternatives_used.append({
                        "fies_id": fies_id,
                        "expected": info.csv_name,
                        "using": info.alternative,
                    })
            else:
                missing.append(fies_id)
        
        return {
            "total_columns": len(self.resolved_columns),
            "found": len(found),
            "missing": len(missing),
            "missing_ids": missing,
            "alternatives_used": alternatives_used,
        }
    
    def rename_to_ids(self) -> pd.DataFrame:
        """
        Return a copy of the DataFrame with columns renamed to FIES IDs.
        
        Useful for standardized data processing.
        """
        rename_map = {}
        for fies_id, info in self.resolved_columns.items():
            if info.found:
                col_name = info.alternative if info.alternative else info.csv_name
                rename_map[col_name] = fies_id
        
        return self.df.rename(columns=rename_map)


def get_column(df: pd.DataFrame, fies_id: str, default=None) -> Optional[pd.Series]:
    """
    Convenience function to get a column by FIES ID.
    
    Args:
        df: pandas DataFrame
        fies_id: FIES variable ID
        default: Value to return if not found
    
    Returns:
        pandas Series or default value
    """
    fies = FIESColumns(df, strict=False)
    return fies.get(fies_id, default=default)


def validate_columns(df: pd.DataFrame, strict: bool = False) -> dict:
    """
    Validate FIES columns in a DataFrame.
    
    Args:
        df: pandas DataFrame
        strict: If True, raise error on missing critical columns
    
    Returns:
        Validation report dict
    
    Raises:
        FIESColumnError: If strict=True and critical columns are missing
    """
    fies = FIESColumns(df, strict=strict)
    report = fies.get_column_report()
    
    missing_critical = fies.validate_critical_columns()
    missing_optional = fies.validate_optional_columns()
    
    report["critical_missing"] = missing_critical
    report["optional_missing"] = missing_optional
    report["is_valid"] = len(missing_critical) == 0
    
    return report


def create_column_aliases(df: pd.DataFrame) -> dict[str, str]:
    """
    Create a mapping of FIES IDs to actual column names in the DataFrame.
    
    Useful for creating flexible column access patterns.
    """
    fies = FIESColumns(df, strict=False)
    aliases = {}
    
    for fies_id, info in fies.resolved_columns.items():
        if info.found:
            col_name = info.alternative if info.alternative else info.csv_name
            aliases[fies_id] = col_name
    
    return aliases
