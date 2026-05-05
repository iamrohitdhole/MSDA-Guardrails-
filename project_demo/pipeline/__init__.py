"""
Offline pipeline scripts: turn raw DrugBank XML into the canonical chatbot
evidence dataset that the runtime reads.

Live runtime never imports anything from this package — these scripts are
build-time only and run from the CLI or Airflow.
"""
