import pandas as pd

file_name = "data/protein_solubility_esol.xlsx"
sheet_name = "Sheet1"

def convert_xl_to_df(file_name, sheet_name) -> pd.DataFrame:
    xl_file = pd.ExcelFile(file_name)
    xl_sheet = xl_file.parse(sheet_name)
    return xl_sheet

if __name__ == "__main__":
    print(convert_xl_to_df(file_name, sheet_name))