import os
import csv
import io
import joblib
import pandas as pd
import streamlit as st

# --------------------------------------------------
# 1. Application configuration
# --------------------------------------------------
st.set_page_config(
    page_title="IT Log Analyzer",
    page_icon="🖥️",
    layout="wide"
)

st.title("🖥️ IT Server Log Analyzer")
st.write(
    "Upload a server log file to analyze original log levels, "
    "detect repeated errors, and compare them with ML predictions."
)

# --------------------------------------------------
# 2. Load the trained Machine Learning model
# --------------------------------------------------
MODEL_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "it_log_model.joblib"
)

@st.cache_resource
def load_model(path):
    return joblib.load(path)

if not os.path.exists(MODEL_PATH):
    st.error(
        f"Model file not found: {MODEL_PATH}. "
        "Make sure it exists in the current Colab runtime."
    )
    st.stop()

try:
    model = load_model(MODEL_PATH)
except Exception as exc:
    st.error(f"Could not load the ML model: {exc}")
    st.stop()

# --------------------------------------------------
# 3. Troubleshooting recommendations
# --------------------------------------------------
recommendations = {
    "Disk space is full": (
        "Check disk usage, remove unnecessary files, "
        "and consider expanding storage after verification."
    ),
    "Network connection failed": (
        "Check connectivity, IP configuration, DNS, "
        "routing, and firewall rules."
    ),
    "CPU usage is high": (
        "Identify processes consuming CPU and review server performance."
    ),
    "Memory usage is high": (
        "Check memory utilization and identify processes "
        "with unusually high memory usage."
    ),
    "Backup failed": (
        "Review backup logs, destination storage, credentials, "
        "and network connectivity."
    )
}

# --------------------------------------------------
# 4. Log parsing helpers
# --------------------------------------------------
LEVEL_MAP = {
    "ERROR": "ERROR",
    "CRITICAL": "ERROR",
    "FATAL": "ERROR",
    "WARNING": "WARNING",
    "WARN": "WARNING",
    "INFORMATION": "INFO",
    "INFO": "INFO",
    "INFORMACIÓN": "INFO",
    "INFORMATIONAL": "INFO",
    "VERBOSE": "INFO",
}

def normalize_level(value):
    value = str(value or "").strip().upper()
    return LEVEL_MAP.get(value, value)

def parse_simple_logs(content):
    """Parse lines such as 'ERROR: Disk space is full'."""
    rows = []
    for raw_line in content.splitlines():
        line = raw_line.strip()
        if not line or ": " not in line:
            continue
        level, message = line.split(": ", 1)
        level = normalize_level(level)
        message = message.strip()
        if level in {"ERROR", "WARNING", "INFO"} and message:
            rows.append({
                "Level": level,
                "Date and Time": "",
                "Source": "",
                "Event ID": "",
                "Task Category": "",
                "Message": message
            })
    return rows

def parse_windows_event_tsv(content):
    """
    Parse tab-delimited Windows Event Viewer exports.
    The common header may omit a Message column even when each record
    contains the message as its final field. csv.reader also supports
    correctly quoted messages containing embedded newlines.
    """
    reader = csv.reader(io.StringIO(content, newline=""), delimiter="\t")
    records = []
    for row in reader:
        if not row or not any(cell.strip() for cell in row):
            continue
        records.append([cell.strip() for cell in row])

    if len(records) < 2:
        return []

    header = [cell.strip().lower() for cell in records[0]]
    if not any("level" in h for h in header) or not any(
        "source" in h for h in header
    ):
        return []

    # Common Event Viewer export columns:
    # Level, Date and Time, Source, Event ID, Task Category, [Message]
    level_idx = next((i for i, h in enumerate(header) if "level" in h), 0)
    date_idx = next((i for i, h in enumerate(header) if "date" in h and "time" in h), 1)
    source_idx = next((i for i, h in enumerate(header) if "source" in h), 2)
    event_idx = next((i for i, h in enumerate(header) if "event id" in h), 3)
    task_idx = next((i for i, h in enumerate(header) if "task categor" in h), 4)
    message_idx = next((i for i, h in enumerate(header) if h == "message"), None)

    rows = []
    for record in records[1:]:
        # Ignore repeated header rows, if any.
        if record and record[0].strip().lower() == "level":
            continue

        def cell(index):
            return record[index].strip() if index < len(record) else ""

        # If the header has no Message column, message is the final field
        # after the standard five columns.
        if message_idx is not None and message_idx < len(record):
            message = record[message_idx].strip()
        elif len(record) > 5:
            message = "\t".join(record[5:]).strip()
        else:
            message = ""

        level = normalize_level(cell(level_idx))
        if level not in {"ERROR", "WARNING", "INFO"}:
            # Preserve unrecognized severities as INFO rather than counting
            # them as errors; original text remains available in Source fields.
            level = "INFO"

        if not message:
            # Some exports can have an empty message. Keep the row visible,
            # but don't invent an event description.
            message = "(No event message included in export)"

        rows.append({
            "Level": level,
            "Date and Time": cell(date_idx),
            "Source": cell(source_idx),
            "Event ID": cell(event_idx),
            "Task Category": cell(task_idx),
            "Message": message
        })

    return rows

# --------------------------------------------------
# 5. Upload and parse a log file
# --------------------------------------------------
uploaded_file = st.file_uploader(
    "Choose a log file",
    type=["log", "txt", "tsv", "csv"]
)

if uploaded_file is not None:
    try:
        content = uploaded_file.getvalue().decode("utf-8-sig", errors="replace")

        # Try Windows Event Viewer tab-separated export first,
        # then the simple LEVEL: message format.
        rows = parse_windows_event_tsv(content)
        file_format = "Windows Event Viewer export" if rows else "LEVEL: message log"
        if not rows:
            rows = parse_simple_logs(content)

        df = pd.DataFrame(
            rows,
            columns=[
                "Level", "Date and Time", "Source",
                "Event ID", "Task Category", "Message"
            ]
        )

        if df.empty:
            st.warning(
                "No valid log entries were found. Supported formats are "
                "'ERROR: Disk space is full' and a tab-delimited Windows "
                "Event Viewer export with Level and Source columns."
            )
            st.stop()

        st.caption(f"Detected format: {file_format}")

        # --------------------------------------------------
        # 6. Machine Learning predictions
        # --------------------------------------------------
        # Predictions are a separate signal; they do not replace the
        # original severity recorded by Windows.
        df["AI Prediction"] = model.predict(df["Message"].fillna("").astype(str).tolist())

        # --------------------------------------------------
        # 7. Summary metrics based on ORIGINAL severity
        # --------------------------------------------------
        errors = df[df["Level"] == "ERROR"]
        warnings = df[df["Level"] == "WARNING"]
        info_logs = df[df["Level"] == "INFO"]

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Logs", len(df))
        col2.metric("Original Errors", len(errors))
        col3.metric("Original Warnings", len(warnings))
        col4.metric("Original Info", len(info_logs))

        # --------------------------------------------------
        # 8. System status based on original severity
        # --------------------------------------------------
        st.subheader("System Status")
        if len(errors) >= 3:
            st.error("HIGH RISK: Multiple events recorded as errors.")
        elif len(errors) >= 1:
            st.warning("ATTENTION REQUIRED: Windows recorded error events.")
        elif len(warnings) >= 1:
            st.warning(
                "REVIEW RECOMMENDED: No error-level events were parsed, "
                "but warning events are present."
            )
        else:
            st.success("No error- or warning-level events were detected in the parsed entries.")

        st.info(
            "AI Prediction is an experimental text classification from a small "
            "training dataset. Use the original Windows event level and event "
            "details for triage; do not treat the prediction as a diagnosis."
        )

        # --------------------------------------------------
        # 9. Error frequency
        # --------------------------------------------------
        st.subheader("Error Frequency")
        if not errors.empty:
            error_summary = (
                errors["Message"]
                .value_counts()
                .rename_axis("Error Message")
                .reset_index(name="Occurrences")
            )
            st.dataframe(error_summary, use_container_width=True)
        else:
            st.info("No error-level entries found in the original log severity.")

        # --------------------------------------------------
        # 10. Troubleshooting recommendations
        # --------------------------------------------------
        st.subheader("Troubleshooting Recommendations")
        if not errors.empty:
            unique_errors = errors["Message"].dropna().unique()
            for message in unique_errors:
                st.markdown(f"**Error event:** {message}")
                recommendation = recommendations.get(
                    message,
                    "Review the event's full message, source, and Event ID. "
                    "Check related events before taking corrective action."
                )
                st.write(f"Suggested action: {recommendation}")
                st.divider()
        elif not warnings.empty:
            st.write(
                "Review warning events and their Event IDs and sources. "
                "A warning is not by itself proof of a security incident."
            )
        else:
            st.success("No error-level entries require investigation based on the parsed severity.")

        # --------------------------------------------------
        # 11. Display all logs and ML predictions
        # --------------------------------------------------
        st.subheader("All Log Entries")
        st.dataframe(df, use_container_width=True)

        st.subheader("AI Prediction Summary")
        prediction_summary = (
            df["AI Prediction"]
            .value_counts()
            .rename_axis("Predicted Category")
            .reset_index(name="Count")
        )
        st.dataframe(prediction_summary, use_container_width=True)

        # Compare original Windows severity with the model output.
        st.subheader("Original Level vs AI Prediction")
        comparison = pd.crosstab(
            df["Level"], df["AI Prediction"], rownames=["Original Level"],
            colnames=["AI Prediction"]
        )
        st.dataframe(comparison, use_container_width=True)

        # --------------------------------------------------
        # 12. Download analysis report
        # --------------------------------------------------
        st.subheader("Download Analysis Report")
        csv_data = df.to_csv(index=False).encode("utf-8-sig")
        st.download_button(
            label="Download Analysis Report (CSV)",
            data=csv_data,
            file_name="it_log_analysis_report.csv",
            mime="text/csv"
        )

    except Exception as exc:
        st.error(f"An error occurred while analyzing the file: {exc}")
