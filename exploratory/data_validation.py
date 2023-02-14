import sys
import click
import mlflow
import pandas as pd
mlflow.autolog()

def carriage_returns(data):
    for index, row in data.iterrows():
        for column, field in row.items():
            try:
                if "\r\n" in field:
                    return index, column, field
            except TypeError:
                continue

def unnamed_columns(data):
    bad_columns = []
    for k in data.keys():
        if "Unnamed" in k:
            bad_columns.append(k)
    return len(bad_columns)

def zero_count_columns(data):
    bad_columns = []
    for k in data.keys():
        if data[k].count() == 0:
            bad_columns.append(k)
    return bad_columns

@click.command()
@click.argument('metrics', type=bool)
@click.argument('max_errors', type=int)
@click.argument('filename', type=click.Path(exists=True))
def main(metrics, max_errors, filename):
    data = pd.read_csv(filename)
    bad_columns = zero_count_columns(data)
    for column in bad_columns:
        click.echo(f"Warning: Column '{column}' has no values")
    unnamed = unnamed_columns(data)

    if unnamed:
        click.echo(f"Warning: found {unnamed} unnamed columns")
    carriage_field = carriage_returns(data)

    if carriage_field:
        index, column, field = carriage_field
        click.echo(
            (
                f"Warning: found carriage returns at index {index}"
                f" of column '{column}':"
            )
        )
        click.echo(f"\t '{field[:50]}'")
    if metrics:
        mlflow.log_metric("unnamed", unnamed)
        mlflow.log_metric("zero_count_columns", len(bad_columns))

if __name__=="__main__":
    main()