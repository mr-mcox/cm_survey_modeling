def add_dimensions(df):
    out = remove_blank_corps(df)
    return out

def melt(df):
    return df

def remove_blank_corps(df):
    return df[df.Corps.notnull()]°