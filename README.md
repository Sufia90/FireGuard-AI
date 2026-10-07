# FireGuard AI

### AI-Powered Fire Safety Insights from Microgravity Combustion Data

**Live Application:** https://9injzxvurbd2nm8nrirgel.streamlit.app

FireGuard AI is an AI-powered research platform designed to help researchers explore NASA's microgravity combustion experiments. It brings BASS-II experimental data and supporting scientific documents into one environment for searching, comparing, analyzing, and explaining experimental evidence.

## Key Features

* **AI Research Assistant:** Ask research questions and explore evidence-backed answers with citations.
* **Experiment Explorer:** Filter and visualize 127 BASS-II experimental tests.
* **Experiment Comparison:** Compare recorded conditions and measurements across two experiments.
* **Machine Learning Analysis:** Predict derived oxygen-depletion proxy bands and explore model probabilities and feature importance.
* **Fire-Safety Insights:** Connect model patterns with supporting experiments and scientific document excerpts.

## Technology Stack

* Python
* Streamlit
* Pandas
* Scikit-learn
* Retrieval-Augmented Generation (RAG)
* Large Language Models (LLMs)
* Data visualization

## NASA Data Source

**BASS-II — NASA Physical Sciences Informatics (PSI)**
https://psi.nasa.gov/physci/repo/data/investigations/PSI-25

The prototype uses a processed table containing 127 BASS-II tests and a BASS-II Science Requirements Document (SRD) PDF as a supporting research source.

## Important Limitations

Machine-learning outputs are predictions of derived oxygen-depletion proxy bands, not official NASA classifications. These predictions are exploratory and should not be interpreted as confirmed scientific findings or direct fire-safety guarantees.

## Project Principle

**Nothing invented, everything cited.**

FireGuard AI aims to make scientific evidence easier to explore while preserving traceability to source material.

## Running Locally

1. Clone or download this repository.
2. Install the required Python packages.
3. Configure any required API keys using environment variables.
4. Run the Streamlit application.

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the application:

```bash
streamlit run app.py
```

Adjust the entry-point filename if your main Streamlit file has a different name.

## Team

**Solo Explorer**
NASA Space Apps Challenge 2026

* Sufia Akter — AI and technical development
* Sadia Farjana Anika — Research, data, and UX/UI

## Acknowledgment

NASA's BASS-II experimental research provides the data foundation for this prototype. NASA remains the source of the referenced research materials; FireGuard AI is an independent project and is not an official NASA product.
