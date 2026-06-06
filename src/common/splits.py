from sklearn.model_selection import train_test_split


def temporal_split(df, time_col="timestamp", train_frac=0.6, val_frac=0.2):
    df = df.sort_values(time_col).reset_index(drop=True)
    n = len(df)
    n_train = int(n * train_frac)
    n_val = int(n * val_frac)
    train = df.iloc[:n_train].copy()
    val = df.iloc[n_train:n_train+n_val].copy()
    test = df.iloc[n_train+n_val:].copy()
    return train, val, test


def random_split(df, target_col, seed=42):
    train, temp = train_test_split(df, test_size=0.4, random_state=seed, stratify=df[target_col])
    val, test = train_test_split(temp, test_size=0.5, random_state=seed, stratify=temp[target_col])
    return train.reset_index(drop=True), val.reset_index(drop=True), test.reset_index(drop=True)
