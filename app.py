"""
app.py
Flask web application: Customer Journey Analytics & Churn Prediction System.
Built on the real UCI Online Retail dataset.

Flow: /login -> /register -> /home -> /predict -> /result -> /history

Run: python3 app.py
Then open: http://127.0.0.1:5000
"""

from flask import Flask, render_template, request, redirect, url_for, session, flash, Response, jsonify, send_file
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
import joblib
import re
import json
import io
import csv
import os
import pandas as pd
import numpy as np
from functools import wraps
from datetime import timedelta, datetime

import database as db

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "churn-prediction-secret-key-change-in-production")
app.permanent_session_lifetime = timedelta(days=7)

EMAIL_REGEX = re.compile(r"^[^\s@]+@[^\s@]+\.[^\s@]+$")

# Initialize the database (creates tables + seeds the demo admin on first run)
db.init_db()

model = joblib.load("models/churn_model.pkl")
scaler = joblib.load("models/scaler.pkl")
feature_columns = joblib.load("models/feature_columns.pkl")
rfm_df = pd.read_csv("data/rfm_features.csv")

with open("models/metrics.json") as f:
    model_metrics = json.load(f)

with open("models/model_comparison.json") as f:
    model_comparison = json.load(f)

# Human-readable descriptions for each feature, used by the explainability
# engine below. Two variants each: one for when the feature is INCREASING
# this customer's churn risk, one for when it's DECREASING it.
FEATURE_EXPLANATIONS = {
    "purchase_rate": {
        "up": "Low order frequency relative to how long they've been a customer",
        "down": "Consistent order pace relative to their tenure",
    },
    "frequency": {
        "up": "Relatively few total orders placed",
        "down": "A strong history of repeat orders",
    },
    "monetary": {
        "up": "Lower total lifetime spend",
        "down": "High total lifetime spend",
    },
    "avg_order_value": {
        "up": "Lower average order value",
        "down": "Higher average order value",
    },
    "tenure_days": {
        "up": "Long tenure without matching recent activity",
        "down": "Shorter customer tenure",
    },
    "clv": {
        "up": "Lower projected customer lifetime value",
        "down": "Higher projected customer lifetime value",
    },
    "F_score": {
        "up": "Below-average order frequency score",
        "down": "Above-average order frequency score",
    },
    "M_score": {
        "up": "Below-average spending score",
        "down": "Above-average spending score",
    },
}


def login_required(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        if "user_id" not in session:
            return redirect(url_for("login"))
        return f(*args, **kwargs)
    return wrapper


def role_required(*allowed_roles):
    """Restrict a route to specific roles (e.g. @role_required('admin'))."""
    def decorator(f):
        @wraps(f)
        def wrapper(*args, **kwargs):
            if "user_id" not in session:
                return redirect(url_for("login"))
            if session.get("role") not in allowed_roles:
                flash("You don't have permission to view that page.", "error")
                return redirect(url_for("home"))
            return f(*args, **kwargs)
        return wrapper
    return decorator


@app.route("/")
def index():
    if "user_id" in session:
        return redirect(url_for("home"))
    return redirect(url_for("login"))


@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "GET":
        return render_template("register.html")

    name = request.form.get("name", "").strip()
    email = request.form.get("email", "").strip().lower()
    password = request.form.get("password", "")
    confirm_password = request.form.get("confirm_password", "")

    errors = []
    if not name or len(name) < 2:
        errors.append("Please enter your full name.")
    if not email or not EMAIL_REGEX.match(email):
        errors.append("Please enter a valid email address.")
    elif db.email_exists(email):
        errors.append("An account with this email already exists. Please sign in instead.")
    if not password or len(password) < 8:
        errors.append("Password must be at least 8 characters long.")
    elif not (re.search(r"[A-Z]", password) and re.search(r"[a-z]", password) and re.search(r"[0-9]", password)):
        errors.append("Password must include an uppercase letter, a lowercase letter, and a number.")
    if password != confirm_password:
        errors.append("Passwords do not match.")

    if errors:
        for e in errors:
            flash(e, "error")
        return render_template("register.html", name=name, email=email)

    user_id = db.create_user(name, email, generate_password_hash(password))
    session.permanent = False
    session["user_id"] = user_id
    session["username"] = name
    session["role"] = "user"
    db.log_activity(user_id, "register", f"New account created: {email}")
    flash("Registration successful. Welcome to Churn Intelligence!", "success")
    return redirect(url_for("home"))


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "GET":
        return render_template("login.html")

    identifier = request.form.get("identifier", "").strip().lower()
    password = request.form.get("password", "")
    remember_me = request.form.get("remember_me") == "on"

    errors = []
    if not identifier:
        errors.append("Email is required.")
    if not password:
        errors.append("Password is required.")

    if errors:
        for e in errors:
            flash(e, "error")
        return render_template("login.html", identifier=identifier)

    user_record = db.get_user_by_email(identifier)

    if user_record is None or not check_password_hash(user_record["password_hash"], password):
        flash("Invalid email or password.", "error")
        return render_template("login.html", identifier=identifier)

    session.permanent = remember_me
    session["user_id"] = user_record["id"]
    session["username"] = user_record["name"]
    session["role"] = user_record["role"]
    db.log_activity(user_record["id"], "login", f"Logged in as {user_record['email']}")
    flash(f"Login successful. Welcome back, {user_record['name']}.", "success")
    return redirect(url_for("home"))


@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))


@app.route("/home")
@login_required
def home():
    stats = {
        "total_customers": len(rfm_df),
        "churn_rate": round(rfm_df["churned"].mean() * 100, 2),
        "avg_clv": round(rfm_df["clv"].mean(), 2),
        "segments": rfm_df["segment"].value_counts().to_dict(),
        "total_transactions": 541909,
    }
    return render_template("home.html", stats=stats, metrics=model_metrics, comparison=model_comparison, username=session.get("username"))


def run_prediction(frequency, monetary, avg_order_value, tenure_days):
    """
    Core prediction logic, shared by the single predict form, the bulk
    CSV upload feature, and the REST API -- so all three paths run
    through the exact same math and can never drift out of sync.
    """
    purchase_rate = frequency / max(tenure_days / 30, 1)
    clv = round(avg_order_value * purchase_rate * 12, 2)

    f_score = 5 if frequency > 15 else 4 if frequency > 8 else 3 if frequency > 4 else 2 if frequency > 1 else 1
    m_score = 5 if monetary > 5000 else 4 if monetary > 2000 else 3 if monetary > 800 else 2 if monetary > 300 else 1

    input_data = pd.DataFrame([{
        "frequency": frequency,
        "monetary": monetary,
        "avg_order_value": avg_order_value,
        "tenure_days": tenure_days,
        "purchase_rate": purchase_rate,
        "clv": clv,
        "F_score": f_score,
        "M_score": m_score
    }])[feature_columns]

    scaled_input = scaler.transform(input_data)
    prediction = int(model.predict(scaled_input)[0])
    probability = float(model.predict_proba(scaled_input)[0][1])
    churn_probability = round(probability * 100, 2)

    if probability > 0.7:
        risk_level = "High"
    elif probability > 0.4:
        risk_level = "Medium"
    else:
        risk_level = "Low"

    value_tier = get_value_tier(clv)

    return {
        "prediction": prediction,
        "churn_prediction": "Likely to Churn" if prediction == 1 else "Likely to Stay",
        "churn_probability": churn_probability,
        "risk_level": risk_level,
        "clv_estimate": clv,
        "value_tier": value_tier,
        "scaled_input": scaled_input[0],
        "inputs": {
            "frequency": frequency,
            "monetary": monetary,
            "avg_order_value": avg_order_value,
            "tenure_days": tenure_days,
        }
    }


@app.route("/predict", methods=["GET", "POST"])
@login_required
def predict():
    if request.method == "GET":
        return render_template("predict.html", username=session.get("username"))

    errors = []
    fields = {}
    for field, label in [
        ("frequency", "Order Frequency"),
        ("monetary", "Total Amount Spent"),
        ("avg_order_value", "Average Order Value"),
        ("tenure_days", "Customer Tenure"),
    ]:
        raw = request.form.get(field, "").strip()
        if not raw:
            errors.append(f"{label} is required.")
            continue
        try:
            value = float(raw)
            if value < 0:
                errors.append(f"{label} cannot be negative.")
            else:
                fields[field] = value
        except ValueError:
            errors.append(f"{label} must be a valid number.")

    if errors:
        for e in errors:
            flash(e, "error")
        return render_template("predict.html", username=session.get("username"), form_data=request.form)

    customer_name = request.form.get("customer_name", "").strip() or None

    result_data = run_prediction(
        fields["frequency"], fields["monetary"], fields["avg_order_value"], fields["tenure_days"]
    )

    recommendations = get_recommendations(result_data["risk_level"], result_data["clv_estimate"])
    explanations = explain_prediction(result_data["scaled_input"])

    # Save this prediction to the user's history
    db.save_prediction(
        user_id=session["user_id"], customer_name=customer_name,
        frequency=fields["frequency"], monetary=fields["monetary"],
        avg_order_value=fields["avg_order_value"], tenure_days=fields["tenure_days"],
        churn_probability=result_data["churn_probability"], risk_level=result_data["risk_level"],
        clv_estimate=result_data["clv_estimate"], value_tier=result_data["value_tier"]
    )
    db.log_activity(session["user_id"], "predict", f"Predicted churn for {customer_name or 'a customer'}: {result_data['risk_level']} risk")

    session["last_result"] = {
        "churn_prediction": result_data["churn_prediction"],
        "churn_probability": result_data["churn_probability"],
        "risk_level": result_data["risk_level"],
        "clv_estimate": result_data["clv_estimate"],
        "value_tier": result_data["value_tier"],
        "recommendations": recommendations,
        "explanations": explanations,
        "customer_name": customer_name,
        "inputs": result_data["inputs"],
    }
    return redirect(url_for("result"))


def explain_prediction(scaled_row, top_n=3):
    """
    Explainability engine for this specific prediction.

    Method: for Logistic Regression, the predicted log-odds is exactly
    intercept + sum(coefficient_i * scaled_feature_value_i). This function
    computes each feature's individual contribution to that sum, which is
    a mathematically exact (not approximated) decomposition of how the
    model reached its answer for THIS customer -- the same underlying
    principle SHAP uses for linear models, without requiring the shap
    library. Positive contribution = pushes toward "churned"; negative
    contribution = pushes toward "retained".

    NOTE: only a subset of features is used as explanation CANDIDATES
    (see STABLE_EXPLANATION_FEATURES below). frequency, monetary, and
    avg_order_value are deliberately excluded here -- they are highly
    correlated with purchase_rate and clv respectively, which causes
    multicollinearity in the linear model: their individual coefficient
    signs can flip counter-intuitively (e.g. raw "frequency" ends up with
    a positive coefficient even though more frequent buyers churn less)
    purely because the correlated feature is absorbing most of the real
    signal. Restricting explanations to features with stable, intuitive
    coefficient signs avoids generating a customer-facing explanation
    that contradicts common sense while still being mathematically
    "correct" for the full model.
    """
    STABLE_EXPLANATION_FEATURES = {"purchase_rate", "clv", "F_score", "M_score"}

    contributions = model.coef_[0] * scaled_row
    candidates = [
        (name, float(contrib)) for name, contrib in zip(feature_columns, contributions)
        if name in STABLE_EXPLANATION_FEATURES
    ]
    ranked = sorted(candidates, key=lambda pair: abs(pair[1]), reverse=True)[:top_n]

    explanations = []
    for feature_name, contribution in ranked:
        direction = "up" if contribution > 0 else "down"
        text = FEATURE_EXPLANATIONS.get(feature_name, {}).get(direction, feature_name)
        explanations.append({
            "text": text,
            "increases_risk": bool(contribution > 0)
        })
    return explanations


def get_value_tier(clv):
    """Classify customer value tier based on estimated CLV, for display purposes."""
    if clv >= 3000:
        return "Gold"
    elif clv >= 1000:
        return "Silver"
    else:
        return "Bronze"


def get_recommendations(risk_level, clv):
    """Business-facing retention recommendations, tailored by BOTH
    churn risk level and customer value (CLV) -- a high-risk, high-value
    customer warrants a very different response than a high-risk,
    low-value one."""
    high_value = clv >= 1500

    if risk_level == "High" and high_value:
        return [
            "URGENT: Assign to a dedicated account manager for immediate, personalized outreach.",
            "Offer an exclusive win-back package (premium discount + free priority shipping).",
            "Escalate to senior customer support to resolve any unaddressed service issues.",
            "Provide early access to new collections as a loyalty gesture before they're lost.",
        ]
    elif risk_level == "High":
        return [
            "Send an automated win-back offer (15-20% discount) within 48 hours.",
            "Trigger a re-engagement email campaign highlighting new arrivals.",
            "Offer free or expedited shipping on their next order to reduce friction.",
            "Add to a low-cost retention email sequence rather than high-touch outreach.",
        ]
    elif risk_level == "Medium" and high_value:
        return [
            "Proactively invite to a premium loyalty/rewards tier.",
            "Assign a personalized product recommendation based on purchase history.",
            "Send a satisfaction check-in survey before any issue escalates.",
            "Offer a meaningful bundle deal to reinforce engagement.",
        ]
    elif risk_level == "Medium":
        return [
            "Invite them to a standard loyalty or rewards program.",
            "Send a satisfaction survey to catch early warning signs.",
            "Offer a bundle deal or cross-sell recommendation based on past purchases.",
            "Send a gentle reminder if their usual purchase window has passed.",
        ]
    else:
        return [
            "Invite them to a referral program — loyal customers are your best acquisition channel.",
            "Offer early access to new product launches or sales events.",
            "Upsell premium or higher-value products aligned with their purchase history.",
            "Request a review or testimonial to strengthen social proof.",
        ]


@app.route("/result")
@login_required
def result():
    result_data = session.get("last_result")
    if not result_data:
        return redirect(url_for("predict"))
    return render_template("result.html", result=result_data, username=session.get("username"))


@app.route("/history")
@login_required
def history():
    risk_filter = request.args.get("risk", "").strip() or None
    search = request.args.get("search", "").strip() or None
    predictions = db.get_predictions_for_user(session["user_id"], risk_filter=risk_filter, search=search)
    return render_template("history.html", predictions=predictions, username=session.get("username"),
                            current_risk=risk_filter or "", current_search=search or "")


# ---------------------------------------------------------------------------
# Bulk CSV upload & prediction
# ---------------------------------------------------------------------------

REQUIRED_BULK_COLUMNS = {"customer_name", "frequency", "monetary", "avg_order_value", "tenure_days"}


@app.route("/bulk-predict", methods=["GET", "POST"])
@login_required
def bulk_predict():
    if request.method == "GET":
        return render_template("bulk_predict.html", username=session.get("username"))

    file = request.files.get("csv_file")
    if not file or file.filename == "":
        flash("Please choose a CSV file to upload.", "error")
        return render_template("bulk_predict.html", username=session.get("username"))

    if not file.filename.lower().endswith(".csv"):
        flash("Only .csv files are supported.", "error")
        return render_template("bulk_predict.html", username=session.get("username"))

    try:
        df = pd.read_csv(file)
    except Exception:
        flash("Couldn't read that file. Make sure it's a valid CSV.", "error")
        return render_template("bulk_predict.html", username=session.get("username"))

    missing_cols = REQUIRED_BULK_COLUMNS - set(df.columns)
    if missing_cols:
        flash(f"CSV is missing required columns: {', '.join(sorted(missing_cols))}", "error")
        return render_template("bulk_predict.html", username=session.get("username"))

    results = []
    errors_count = 0
    for _, row in df.iterrows():
        try:
            frequency = float(row["frequency"])
            monetary = float(row["monetary"])
            avg_order_value = float(row["avg_order_value"])
            tenure_days = float(row["tenure_days"])
            customer_name = str(row["customer_name"]) if pd.notna(row["customer_name"]) else "Unknown"

            if frequency < 0 or monetary < 0 or avg_order_value < 0 or tenure_days < 0:
                errors_count += 1
                continue

            r = run_prediction(frequency, monetary, avg_order_value, tenure_days)

            db.save_prediction(
                user_id=session["user_id"], customer_name=customer_name,
                frequency=frequency, monetary=monetary, avg_order_value=avg_order_value,
                tenure_days=tenure_days, churn_probability=r["churn_probability"],
                risk_level=r["risk_level"], clv_estimate=r["clv_estimate"], value_tier=r["value_tier"]
            )

            results.append({
                "customer_name": customer_name,
                "frequency": frequency, "monetary": monetary,
                "avg_order_value": avg_order_value, "tenure_days": tenure_days,
                "churn_probability": r["churn_probability"], "risk_level": r["risk_level"],
                "clv_estimate": r["clv_estimate"], "value_tier": r["value_tier"],
            })
        except (ValueError, KeyError, TypeError):
            errors_count += 1
            continue

    db.log_activity(session["user_id"], "bulk_upload", f"Uploaded CSV, predicted {len(results)} customers ({errors_count} skipped)")

    if not results:
        flash("No valid rows could be processed from that file.", "error")
        return render_template("bulk_predict.html", username=session.get("username"))

    session["bulk_results"] = results
    flash(f"Processed {len(results)} customers successfully" + (f" ({errors_count} rows skipped due to errors)" if errors_count else "") + ".", "success")
    return render_template("bulk_predict.html", username=session.get("username"), results=results)


@app.route("/bulk-predict/download")
@login_required
def bulk_predict_download():
    results = session.get("bulk_results")
    if not results:
        flash("No bulk results to download. Run a bulk prediction first.", "error")
        return redirect(url_for("bulk_predict"))

    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=list(results[0].keys()))
    writer.writeheader()
    writer.writerows(results)

    return Response(
        output.getvalue(),
        mimetype="text/csv",
        headers={"Content-Disposition": "attachment; filename=bulk_prediction_results.csv"}
    )


# ---------------------------------------------------------------------------
# Export prediction history (CSV / Excel / PDF)
# ---------------------------------------------------------------------------

@app.route("/export/csv")
@login_required
def export_csv():
    predictions = db.get_predictions_for_user(session["user_id"], limit=10000)
    output = io.StringIO()
    if predictions:
        writer = csv.DictWriter(output, fieldnames=list(predictions[0].keys()))
        writer.writeheader()
        writer.writerows(predictions)
    db.log_activity(session["user_id"], "export", "Exported prediction history as CSV")
    return Response(
        output.getvalue(),
        mimetype="text/csv",
        headers={"Content-Disposition": "attachment; filename=prediction_history.csv"}
    )


@app.route("/export/excel")
@login_required
def export_excel():
    predictions = db.get_predictions_for_user(session["user_id"], limit=10000)
    df = pd.DataFrame(predictions) if predictions else pd.DataFrame(
        columns=["id", "customer_name", "frequency", "monetary", "avg_order_value",
                 "tenure_days", "churn_probability", "risk_level", "clv_estimate", "value_tier", "created_at"]
    )
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="Prediction History")
    output.seek(0)
    db.log_activity(session["user_id"], "export", "Exported prediction history as Excel")
    return send_file(
        output, mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        as_attachment=True, download_name="prediction_history.xlsx"
    )


@app.route("/export/pdf")
@login_required
def export_pdf():
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import landscape, A4
    from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.lib.units import cm

    predictions = db.get_predictions_for_user(session["user_id"], limit=1000)

    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=landscape(A4), topMargin=1.5*cm, bottomMargin=1.5*cm)
    styles = getSampleStyleSheet()
    elements = [Paragraph("Churn Prediction History", styles["Title"]), Spacer(1, 0.5*cm)]

    header = ["Customer", "Frequency", "Spent (£)", "Tenure", "Churn %", "Risk", "Tier", "CLV (£)", "Date"]
    data = [header]
    for p in predictions:
        data.append([
            p.get("customer_name") or "-", p["frequency"], f"{p['monetary']:.0f}",
            f"{p['tenure_days']:.0f}d", f"{p['churn_probability']}%", p["risk_level"],
            p["value_tier"], f"{p['clv_estimate']:.0f}", p["created_at"][:10]
        ])

    table = Table(data, repeatRows=1)
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1a2a4a")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTSIZE", (0, 0), (-1, -1), 8),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f4f6f9")]),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ]))
    elements.append(table)
    doc.build(elements)
    buffer.seek(0)

    db.log_activity(session["user_id"], "export", "Exported prediction history as PDF")
    return send_file(buffer, mimetype="application/pdf", as_attachment=True, download_name="prediction_history.pdf")


# ---------------------------------------------------------------------------
# User profile
# ---------------------------------------------------------------------------

@app.route("/profile")
@login_required
def profile():
    user = db.get_user_by_id(session["user_id"])
    stats = db.get_prediction_stats_for_user(session["user_id"])
    return render_template("profile.html", user=user, stats=stats, username=session.get("username"))


# ---------------------------------------------------------------------------
# Forgot / reset password
# ---------------------------------------------------------------------------

@app.route("/forgot-password", methods=["GET", "POST"])
def forgot_password():
    if request.method == "GET":
        return render_template("forgot_password.html")

    email = request.form.get("email", "").strip()
    user = db.get_user_by_email(email)

    # NOTE: no real email service is configured in this environment, so
    # instead of emailing the reset link, it is shown directly on the page.
    # In production this token would be sent via email (e.g. Flask-Mail)
    # and never displayed in the browser.
    if user:
        token = db.create_reset_token(user["id"])
        reset_link = url_for("reset_password", token=token, _external=True)
        db.log_activity(user["id"], "password_reset_requested", f"Reset requested for {email}")
        return render_template("forgot_password.html", reset_link=reset_link, email=email)

    # Don't reveal whether the email exists, for basic security hygiene
    flash("If that email is registered, a reset link has been generated below.", "success")
    return render_template("forgot_password.html")


@app.route("/reset-password/<token>", methods=["GET", "POST"])
def reset_password(token):
    user_id = db.validate_reset_token(token)
    if not user_id:
        flash("This reset link is invalid or has expired. Please request a new one.", "error")
        return redirect(url_for("forgot_password"))

    if request.method == "GET":
        return render_template("reset_password.html", token=token)

    password = request.form.get("password", "")
    confirm_password = request.form.get("confirm_password", "")

    errors = []
    if len(password) < 8:
        errors.append("Password must be at least 8 characters.")
    if not re.search(r"[A-Z]", password) or not re.search(r"[a-z]", password) or not re.search(r"\d", password):
        errors.append("Password must include uppercase, lowercase, and a number.")
    if password != confirm_password:
        errors.append("Passwords do not match.")

    if errors:
        for e in errors:
            flash(e, "error")
        return render_template("reset_password.html", token=token)

    db.update_password(user_id, generate_password_hash(password))
    db.mark_token_used(token)
    db.log_activity(user_id, "password_reset_completed", "Password was reset")
    flash("Password reset successfully. You can now log in.", "success")
    return redirect(url_for("login"))


# ---------------------------------------------------------------------------
# Admin: activity logs
# ---------------------------------------------------------------------------

@app.route("/admin/logs")
@role_required("admin")
def admin_logs():
    logs = db.get_all_activity_logs()
    return render_template("admin_logs.html", logs=logs, username=session.get("username"))


# ---------------------------------------------------------------------------
# REST API
# ---------------------------------------------------------------------------

@app.route("/api/predict", methods=["POST"])
def api_predict():
    """
    JSON API for churn prediction, so other applications can integrate
    without needing the web UI.

    Example request body:
    {"frequency": 5, "monetary": 1200, "avg_order_value": 240, "tenure_days": 250}
    """
    data = request.get_json(silent=True)
    if not data:
        return jsonify({"error": "Request body must be JSON."}), 400

    required = ["frequency", "monetary", "avg_order_value", "tenure_days"]
    missing = [f for f in required if f not in data]
    if missing:
        return jsonify({"error": f"Missing required fields: {missing}"}), 400

    try:
        frequency = float(data["frequency"])
        monetary = float(data["monetary"])
        avg_order_value = float(data["avg_order_value"])
        tenure_days = float(data["tenure_days"])
    except (ValueError, TypeError):
        return jsonify({"error": "All fields must be numeric."}), 400

    if any(v < 0 for v in [frequency, monetary, avg_order_value, tenure_days]):
        return jsonify({"error": "Values cannot be negative."}), 400

    r = run_prediction(frequency, monetary, avg_order_value, tenure_days)
    return jsonify({
        "churn_prediction": r["churn_prediction"],
        "churn_probability": r["churn_probability"],
        "risk_level": r["risk_level"],
        "clv_estimate": r["clv_estimate"],
        "value_tier": r["value_tier"],
    })


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    debug_mode = os.environ.get("FLASK_DEBUG", "true").lower() == "true"
    app.run(debug=debug_mode, host="0.0.0.0", port=port)
