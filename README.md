# AI-Powered Customer Churn Prediction and Customer Retention Analytics Platform

## Project Live Deployment Link:
https://ai-powered-customer-churn-prediction-and.onrender.com/login

A full-stack, production-shaped churn prediction platform built on the
**real UCI Online Retail dataset** (541,910 transactions, UK-based online
gift retailer, Dec 2010 – Dec 2011). RFM segmentation, CLV estimation,
a Logistic Regression model, a SQLite-backed multi-user system with
registration, bulk prediction, model explainability, exports, an admin
audit trail, a REST API, and a glassmorphism UI with dark/light themes.

## Project Structure
```
churn_project_v2/
├── rfm_analysis.py          # Loads + cleans real data, RFM + CLV + churn labeling
├── churn_model.py           # Trains/compares models, saves final model + comparison metrics
├── database.py              # SQLite layer: users, predictions, activity_logs, reset tokens
├── churn_analysis.ipynb     # Full EDA + modeling notebook (pre-run, portfolio-ready)
├── app.py                   # Flask app — all routes, prediction logic, REST API
├── templates/
│   ├── register.html         # Account creation
│   ├── login.html            # Login + "Forgot password?" link
│   ├── forgot_password.html  # Request a reset link
│   ├── reset_password.html   # Set a new password
│   ├── home.html              # Dashboard: stats, model comparison + chart, RFM segments
│   ├── predict.html          # Single-customer RFM prediction form
│   ├── bulk_predict.html     # CSV upload → predict many customers at once
│   ├── result.html           # Churn gauge, explainability, recommendations
│   ├── history.html          # "My Predictions" with search/filter + exports
│   ├── profile.html          # Account details + prediction activity stats
│   └── admin_logs.html       # Admin-only activity audit trail
├── static/css/
│   ├── login.css / register.css   # Auth page styling
│   ├── home.css / predict.css / result.css   # Core 3-page styling
│   └── app.css                # Shared styling for history/bulk/profile/logs (no duplication)
├── data/                     # rfm_features.csv lives here
├── models/                   # Trained model + metrics + comparison JSON
├── database.db                # SQLite database (created on first run)
├── Procfile                  # For deployment (Render/Railway/Heroku-style)
└── requirements.txt
```

## How to Run Locally

```bash
cd churn_project_v2
pip install -r requirements.txt

python3 rfm_analysis.py       # generates data/rfm_features.csv
python3 churn_model.py        # trains model, saves comparison metrics
python3 app.py                # starts the server, creates database.db on first run
```

### Demo Login
```
Email:    admin@churnapp.com
Password: Admin@123
```
Or register your own account via the login page.

## Full Feature List

### Core Prediction
- RFM-based churn prediction (Logistic Regression, ~87.5% accuracy)
- Data-leakage-free feature set (recency/R_score excluded — see notebook Section 10)
- **Model explainability** — top 3 factors behind each prediction, with an
  honest handling of a multicollinearity edge case (see below)
- **Recommendation engine** — tailored by both risk level AND customer value (CLV)

### User System
- Registration with live password-strength validation
- Login with "Remember me" and **Forgot/Reset Password** (token-based; since
  no email server is configured, the reset link is shown on-screen instead
  of emailed — see the note in `forgot_password.html`)
- **Role-based access** (`user` / `admin`) — admin-only Activity Logs page
- User **Profile** page with prediction activity stats

### Bulk & Export
- **Bulk CSV upload** — predict churn for many customers at once, view
  results in-browser, download as CSV
- **Export prediction history** as CSV, Excel (`.xlsx`), or PDF

### Search & History
- Full prediction history per user, persisted in SQLite
- **Search** by customer name/ID and **filter** by risk level

### Dashboard
- Live stats: total customers, churn rate, average CLV
- **Full model comparison table** (Logistic Regression vs Random Forest vs
  Gradient Boosting) with an **interactive Chart.js bar chart**
- RFM segment breakdown with animated bars
- Retention donut chart + key insight summary

### Platform
- **Dark/Light theme toggle** (persisted via localStorage, no flash-of-wrong-theme)
- **REST API** (`POST /api/predict`) for external integration
- **Admin activity audit log** — logins, predictions, uploads, exports, password resets
- Responsive design (mobile breakpoints across all pages)
- Deployment live on Render

## Database — Real, Persistent Storage (SQLite)

```
users                    predictions              activity_logs         password_reset_tokens
├── id                   ├── id                    ├── id                 ├── id
├── name                 ├── user_id (FK)           ├── user_id (FK)       ├── user_id (FK)
├── email                ├── customer_name          ├── action             ├── token
├── password_hash        ├── frequency, monetary...  ├── details            ├── expires_at
├── role                 ├── churn_probability       └── created_at        ├── used
└── created_at           ├── risk_level, value_tier                       └── created_at
                          └── created_at
```

**Why SQLite instead of MySQL:** no server setup required — a single file,
built into Python. The SQL is standard ANSI SQL; migrating to MySQL later
means swapping the connection method and one line of table syntax
(`AUTOINCREMENT` → `AUTO_INCREMENT`). Application logic wouldn't need to change.


## Limitations

- **Password reset emails aren't actually sent** — no SMTP/email service is
  configured in this environment. The reset link is displayed on-screen
  instead. In production, this would go through a real email service
  (e.g. Flask-Mail + a provider like SendGrid).
- **Email notifications for high-risk customers** were not implemented —
  same email-service limitation.
- **XGBoost was not added** to the model comparison — the environment
  used to build this had no internet access to install it. The comparison
  currently covers Logistic Regression, Random Forest, and Gradient Boosting.
 
## Project Highlights

1. **Real transactional data**, not synthetic — 541K+ rows, honestly cleaned
2. **Data leakage explicitly prevented and explained**
3. **Full model comparison**, not just the winning model
4. **Genuine explainability** with an honest multicollinearity caveat
5. **Real multi-user system** — registration, roles, persistent database
6. **Bulk processing + exports** — the kind of feature real internal tools need
7. **Admin audit trail** — shows understanding of production concerns
8. **REST API** — shows the model can be consumed by other systems
9. **Deployment live** , not just running locally   


## Tech Stack

- Python
- Flask
- Scikit-learn
- Pandas
- NumPy
- SQLite
- HTML5
- CSS3
- JavaScript
- Chart.js
- Bootstrap
- Git
- GitHub

## Model Performance

- Logistic Regression Accuracy: 87.56%
- Cross Validation Accuracy: 86.17%
- Dataset Size: 541,909 Transactions
- Customers: 4,338

## Conclusion

This project demonstrates a complete, production-shaped machine learning system — not just a model in a notebook, but a deployed, usable application. Starting from a real, messy 541,000-row dataset, it covers the full lifecycle: data cleaning, feature engineering (RFM/CLV), model selection with explicit data-leakage prevention, algorithm comparison, model explainability, and a full-stack multi-user web application with authentication, bulk processing, exports, and a REST API — deployed live and accessible to anyone. The goal throughout was honesty over impressiveness — every accuracy number reported here is real and reproducible, and every limitation is documented rather than hidden.
