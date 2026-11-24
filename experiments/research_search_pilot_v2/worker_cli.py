"""Bind the missing control exception; delegate unchanged scientific execution."""
from research_search.orchestration.budget import BudgetError


def bind_control_exception():
    from research_search.runners import train
    train.BudgetError=BudgetError


def main():
    bind_control_exception()
    from research_search.runners.evaluate import main as native_main
    native_main()

if __name__=='__main__':main()
