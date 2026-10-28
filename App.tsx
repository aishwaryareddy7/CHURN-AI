import {
  useEffect,
  useState,
  type Dispatch,
  type ReactNode,
  type SetStateAction,
} from "react";

import {
  Activity,
  BarChart3,
  Brain,
  ChevronLeft,
  ChevronRight,
  Database,
  Gauge,
  Info,
  LayoutDashboard,
  Menu,
  Network,
  Search,
  ShieldCheck,
  Sparkles,
  Target,
  Users,
  X,
  Zap,
  ArrowUpRight,
  CheckCircle2,
  CircleDot,
} from "lucide-react";

import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  Legend,
  Line,
  Pie,
  PieChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

const API = "http://127.0.0.1:8000";

type Page =
  | "dashboard"
  | "predict"
  | "explanation"
  | "analytics"
  | "performance"
  | "xai"
  | "dataset"
  | "about";

type ModelResult = {
  name: string;
  role: string;
  status: string;
  accuracy?: number;
  precision?: number;
  recall?: number;
  f1_score?: number;
  roc_auc?: number;
  pr_auc?: number;
  confusion_matrix?: {
    true_negative: number;
    false_positive: number;
    false_negative: number;
    true_positive: number;
  };
  error?: string;
};

type DatasetSummary = {
  total_customers: number;
  total_features: number;
  total_columns: number;
  churned_customers: number;
  retained_customers: number;
  churn_rate: number;
};

type ModelStatus = {
  primary_model: string;
  backup_model: string;
  active_model: string;
  tabpfn_available: boolean;
  dataset_rows: number;
  dataset_columns: number;
  feature_count: number;
};

type PredictionResult = {
  churn_probability: number;
  churn_percentage: number;
  risk_level: string;
  model_used: string;
  recommendation: string;
};

type ShapFactor = {
  feature: string;
  name: string;
  value: number;
  shap_value: number;
  absolute_impact: number;
  direction: "increases" | "decreases";
  impact: string;
};

type ExplanationResult = {
  success: boolean;
  model_used: string;
  churn_probability: number;
  churn_percentage: number;
  risk_level: string;
  top_factors: ShapFactor[];
  positive_factors: ShapFactor[];
  negative_factors: ShapFactor[];
  explanation_method: string;
  background_samples: number;
  note: string;
};

type AnalyticsData = {
  contract_churn: Record<string, number>;
  internet_service_churn: Record<string, number>;
  payment_method_churn: Record<string, number>;
  senior_citizen_churn: Record<string, number>;
};

type DatasetResponse = {
  data: Record<string, string | number | null>[];
  columns: string[];
  page: number;
  page_size: number;
  total_rows: number;
  total_pages: number;
  search: string;
};

const initialCustomer = {
  gender: "Male",
  senior_citizen: 0,
  partner: 1,
  dependents: 0,
  tenure: 12,
  contract: "Month-to-month",
  paperless_billing: 1,
  payment_method: "Electronic check",
  monthly_charges: 80,
  total_charges: 960,
  phone_service: 1,
  multiple_lines: 0,
  internet_service: "Fiber optic",
  online_security: 0,
  online_backup: 0,
  device_protection: 0,
  tech_support: 0,
  streaming_tv: 1,
  streaming_movies: 1,
};

function App() {
  const [page, setPage] = useState<Page>("dashboard");
  const [sidebarOpen, setSidebarOpen] = useState(false);

  const [summary, setSummary] =
    useState<DatasetSummary | null>(null);

  const [modelStatus, setModelStatus] =
    useState<ModelStatus | null>(null);

  const [prediction, setPrediction] =
    useState<PredictionResult | null>(null);

  const [explanation, setExplanation] =
    useState<ExplanationResult | null>(null);

  const [explaining, setExplaining] =
    useState(false);

  const [customer, setCustomer] =
    useState(initialCustomer);

  const [predicting, setPredicting] =
    useState(false);

  const [analytics, setAnalytics] =
    useState<AnalyticsData | null>(null);

  const [modelResults, setModelResults] =
    useState<ModelResult[]>([]);

  const [modelLoading, setModelLoading] =
    useState(false);

  const [dataset, setDataset] =
    useState<DatasetResponse | null>(null);

  const [datasetLoading, setDatasetLoading] =
    useState(false);

  const [datasetPage, setDatasetPage] =
    useState(1);

  const [datasetSearch, setDatasetSearch] =
    useState("");

  useEffect(() => {
    fetch(`${API}/api/dataset-summary`)
      .then((response) => response.json())
      .then(setSummary)
      .catch(console.error);

    fetch(`${API}/api/model-status`)
      .then((response) => response.json())
      .then(setModelStatus)
      .catch(console.error);

    fetch(`${API}/api/analytics`)
      .then((response) => response.json())
      .then(setAnalytics)
      .catch(console.error);
  }, []);

  useEffect(() => {
    if (page !== "performance") {
      return;
    }

    setModelLoading(true);

    fetch(`${API}/api/model-comparison`)
      .then((response) => response.json())
      .then((data) => {
        setModelResults(data.models || []);
      })
      .catch(console.error)
      .finally(() => {
        setModelLoading(false);
      });
  }, [page]);

  useEffect(() => {
    if (page !== "dataset") {
      return;
    }

    setDatasetLoading(true);

    const params = new URLSearchParams({
      page: String(datasetPage),
      page_size: "50",
      search: datasetSearch,
    });

    fetch(`${API}/api/dataset?${params.toString()}`)
      .then((response) => response.json())
      .then(setDataset)
      .catch(console.error)
      .finally(() => {
        setDatasetLoading(false);
      });
  }, [page, datasetPage, datasetSearch]);

  const navigate = (nextPage: Page) => {
    setPage(nextPage);
    setSidebarOpen(false);
  };

  const runPrediction = async () => {
    setPredicting(true);
    setExplanation(null);

    try {
      const response = await fetch(
        `${API}/api/predict`,
        {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify(customer),
        }
      );

      const data = await response.json();

      if (!response.ok) {
        console.error("Prediction failed:", data);
        return;
      }

      setPrediction(data);
      setExplaining(true);

      try {
        const explanationResponse = await fetch(
          `${API}/api/explain`,
          {
            method: "POST",
            headers: {
              "Content-Type": "application/json",
            },
            body: JSON.stringify(customer),
          }
        );

        const explanationData =
          await explanationResponse.json();

        if (explanationResponse.ok) {
          setExplanation(explanationData);
        } else {
          console.error(
            "SHAP explanation failed:",
            explanationData
          );
        }
      } catch (error) {
        console.error(
          "SHAP explanation error:",
          error
        );
      } finally {
        setExplaining(false);
      }
    } catch (error) {
      console.error(error);
    } finally {
      setPredicting(false);
    }
  };

  const updateCustomer = (
    key: keyof typeof initialCustomer,
    value: string | number
  ) => {
    setCustomer((previous) => ({
      ...previous,
      [key]: value,
    }));
  };

  return (
    <div className="min-h-screen bg-[#060914] text-white">

      <Sidebar
        page={page}
        navigate={navigate}
        open={sidebarOpen}
        setOpen={setSidebarOpen}
        modelStatus={modelStatus}
      />

      <div className="lg:pl-72">

        <header className="sticky top-0 z-30 border-b border-white/10 bg-[#060914]/90 backdrop-blur-xl">

          <div className="flex h-16 items-center justify-between px-4 sm:px-6 lg:px-8">

            <button
              onClick={() => setSidebarOpen(true)}
              className="rounded-lg border border-white/10 p-2 lg:hidden"
            >
              <Menu size={20} />
            </button>

            <div className="hidden lg:block">
              <p className="text-xs uppercase tracking-[0.25em] text-slate-500">
                Telecom Intelligence Platform
              </p>

              <h1 className="text-sm font-semibold text-slate-200">
                Customer Churn Prediction
              </h1>
            </div>

            <div className="flex items-center gap-3">

              <div className="hidden sm:flex items-center gap-2 rounded-full border border-blue-400/20 bg-blue-400/5 px-3 py-1.5 text-xs text-blue-300">
                <CircleDot size={13} />
                AI Engine Active
              </div>

              <div className="flex items-center gap-2 rounded-full border border-emerald-400/20 bg-emerald-400/5 px-3 py-1.5 text-xs text-emerald-300">
                <span className="h-2 w-2 rounded-full bg-emerald-400 shadow-[0_0_10px_rgba(52,211,153,0.8)]" />
                System Online
              </div>

            </div>

          </div>

        </header>

        <main className="p-4 sm:p-6 lg:p-8">

          {page === "dashboard" && (
            <Dashboard
              summary={summary}
              modelStatus={modelStatus}
              navigate={navigate}
            />
          )}

          {page === "predict" && (
            <PredictPage
              customer={customer}
              updateCustomer={updateCustomer}
              prediction={prediction}
              predicting={predicting}
              runPrediction={runPrediction}
            />
          )}

          {page === "explanation" && (
            <ExplanationPage
              prediction={prediction}
              explanation={explanation}
              explaining={explaining}
              navigate={navigate}
            />
          )}

          {page === "analytics" && (
            <AnalyticsPage analytics={analytics} />
          )}

          {page === "performance" && (
            <PerformancePage
              results={modelResults}
              loading={modelLoading}
              modelStatus={modelStatus}
            />
          )}

          {page === "xai" && <XAIPage />}

          {page === "dataset" && (
            <DatasetPage
              dataset={dataset}
              loading={datasetLoading}
              page={datasetPage}
              setPage={setDatasetPage}
              search={datasetSearch}
              setSearch={(value) => {
                setDatasetPage(1);
                setDatasetSearch(value);
              }}
            />
          )}

          {page === "about" && <AboutPage />}

        </main>
      </div>
    </div>
  );
}


/* ============================================================
   SIDEBAR
============================================================ */

function Sidebar({
  page,
  navigate,
  open,
  setOpen,
  modelStatus,
}: {
  page: Page;
  navigate: (page: Page) => void;
  open: boolean;
  setOpen: Dispatch<SetStateAction<boolean>>;
  modelStatus: ModelStatus | null;
}) {
  const items: {
    id: Page;
    label: string;
    icon: ReactNode;
  }[] = [
    {
      id: "dashboard",
      label: "Dashboard",
      icon: <LayoutDashboard size={18} />,
    },
    {
      id: "predict",
      label: "Predict Churn",
      icon: <Target size={18} />,
    },
    {
      id: "explanation",
      label: "Customer Explanation",
      icon: <Sparkles size={18} />,
    },
    {
      id: "analytics",
      label: "Analytics",
      icon: <BarChart3 size={18} />,
    },
    {
      id: "performance",
      label: "Model Performance",
      icon: <Gauge size={18} />,
    },
    {
      id: "xai",
      label: "Explainable AI",
      icon: <Brain size={18} />,
    },
    {
      id: "dataset",
      label: "Full Dataset",
      icon: <Database size={18} />,
    },
    {
      id: "about",
      label: "About Project",
      icon: <Info size={18} />,
    },
  ];

  return (
    <>
      {open && (
        <div
          className="fixed inset-0 z-40 bg-black/60 lg:hidden"
          onClick={() => setOpen(false)}
        />
      )}

      <aside
        className={`
          fixed inset-y-0 left-0 z-50 w-72
          border-r border-white/10
          bg-[#080b17]
          transition-transform duration-300
          lg:translate-x-0
          ${open ? "translate-x-0" : "-translate-x-full"}
        `}
      >
        <div className="flex h-full flex-col">

          <div className="flex items-center justify-between border-b border-white/10 px-6 py-5">

            <div className="flex items-center gap-3">

              <div className="relative flex h-10 w-10 items-center justify-center rounded-xl bg-gradient-to-br from-blue-500 to-violet-600 shadow-lg shadow-blue-500/20">
                <Network size={21} />

                <span className="absolute -right-1 -top-1 h-2.5 w-2.5 rounded-full border-2 border-[#080b17] bg-emerald-400" />
              </div>

              <div>
                <div className="text-lg font-bold tracking-tight">
                  CHURNIQ
                </div>

                <div className="text-[10px] uppercase tracking-[0.25em] text-slate-500">
                  AI Retention Engine
                </div>
              </div>

            </div>

            <button
              onClick={() => setOpen(false)}
              className="rounded-lg p-2 text-slate-500 hover:bg-white/5 lg:hidden"
            >
              <X size={18} />
            </button>

          </div>

          <nav className="flex-1 space-y-1 overflow-y-auto p-4">

            {items.map((item) => (
              <button
                key={item.id}
                onClick={() => navigate(item.id)}
                className={`
                  flex w-full items-center gap-3 rounded-xl px-4 py-3
                  text-sm transition
                  ${
                    page === item.id
                      ? "bg-blue-500/10 text-blue-300 ring-1 ring-blue-400/20"
                      : "text-slate-400 hover:bg-white/5 hover:text-white"
                  }
                `}
              >
                {item.icon}
                <span>{item.label}</span>

                {page === item.id && (
                  <span className="ml-auto h-1.5 w-1.5 rounded-full bg-blue-400 shadow-[0_0_8px_rgba(96,165,250,0.8)]" />
                )}
              </button>
            ))}

          </nav>

          <div className="border-t border-white/10 p-4">

            <div className="rounded-xl border border-white/10 bg-white/[0.03] p-4">

              <div className="mb-3 flex items-center gap-2 text-xs font-semibold text-slate-300">

                <ShieldCheck
                  size={15}
                  className="text-emerald-400"
                />

                Model Status

              </div>

              <div className="flex items-center justify-between">

                <span className="text-xs text-slate-500">
                  Active Model
                </span>

                <span className="text-xs font-semibold text-blue-300">
                  {modelStatus?.active_model || "Loading..."}
                </span>

              </div>

              <div className="mt-2 flex items-center justify-between">

                <span className="text-xs text-slate-500">
                  TabPFN
                </span>

                <span
                  className={
                    modelStatus?.tabpfn_available
                      ? "text-xs text-emerald-400"
                      : "text-xs text-amber-400"
                  }
                >
                  {modelStatus?.tabpfn_available
                    ? "Available"
                    : "Fallback"}
                </span>

              </div>

            </div>

          </div>

        </div>
      </aside>
    </>
  );
}


/* ============================================================
   DASHBOARD — POLISHED
============================================================ */

function Dashboard({
  summary,
  modelStatus,
  navigate,
}: {
  summary: DatasetSummary | null;
  modelStatus: ModelStatus | null;
  navigate: (page: Page) => void;
}) {
  const churnData = summary
    ? [
        {
          name: "Retained",
          value: summary.retained_customers,
        },
        {
          name: "Churned",
          value: summary.churned_customers,
        },
      ]
    : [];

  const churnPercentage = summary
    ? summary.churn_rate
    : 0;

  return (
    <div className="space-y-7">

      {/* HERO */}

      <section className="relative overflow-hidden rounded-3xl border border-blue-400/10 bg-gradient-to-br from-blue-500/[0.10] via-violet-500/[0.06] to-transparent p-6 sm:p-8 lg:p-10">

        <div className="absolute -right-20 -top-24 h-72 w-72 rounded-full bg-blue-500/10 blur-3xl" />

        <div className="absolute -bottom-28 right-32 h-64 w-64 rounded-full bg-violet-500/10 blur-3xl" />

        <div className="relative">

          <div className="flex flex-col gap-8 lg:flex-row lg:items-end lg:justify-between">

            <div className="max-w-3xl">

              <div className="mb-4 inline-flex items-center gap-2 rounded-full border border-blue-400/20 bg-blue-400/5 px-3 py-1.5 text-xs font-medium text-blue-300">
                <Sparkles size={13} />
                AI-Powered Retention Intelligence
              </div>

              <h1 className="text-3xl font-bold tracking-tight sm:text-4xl lg:text-5xl">
                Predict churn.
                <span className="block bg-gradient-to-r from-blue-300 via-violet-300 to-blue-400 bg-clip-text text-transparent">
                  Understand why.
                </span>
              </h1>

              <p className="mt-4 max-w-2xl text-sm leading-7 text-slate-400 sm:text-base">
                CHURNIQ combines customer data, TabPFN prediction,
                and SHAP explainability to transform churn signals
                into actionable retention decisions.
              </p>

            </div>

            <div className="shrink-0">

              <button
                onClick={() => navigate("predict")}
                className="group flex items-center gap-3 rounded-xl bg-gradient-to-r from-blue-500 to-violet-600 px-5 py-3.5 text-sm font-semibold shadow-lg shadow-blue-500/20 transition hover:scale-[1.02] hover:brightness-110"
              >
                <Target size={18} />

                Predict a Customer

                <ArrowUpRight
                  size={17}
                  className="transition group-hover:translate-x-0.5 group-hover:-translate-y-0.5"
                />
              </button>

            </div>

          </div>

          <div className="mt-8 flex flex-wrap items-center gap-x-6 gap-y-3 border-t border-white/10 pt-5">

            <div className="flex items-center gap-2 text-xs text-slate-400">
              <CheckCircle2
                size={14}
                className="text-emerald-400"
              />
              Real-time prediction
            </div>

            <div className="flex items-center gap-2 text-xs text-slate-400">
              <CheckCircle2
                size={14}
                className="text-emerald-400"
              />
              SHAP explanations
            </div>

            <div className="flex items-center gap-2 text-xs text-slate-400">
              <CheckCircle2
                size={14}
                className="text-emerald-400"
              />
              Actionable recommendations
            </div>

          </div>

        </div>

      </section>


      {/* KPI CARDS */}

      <div className="grid gap-4 sm:grid-cols-2 xl:grid-cols-4">

        <StatCard
          icon={<Users size={19} />}
          label="Total Customers"
          value={
            summary
              ? summary.total_customers.toLocaleString()
              : "—"
          }
          accent="blue"
          subtitle="Customer records analyzed"
        />

        <StatCard
          icon={<Activity size={19} />}
          label="Churn Rate"
          value={
            summary
              ? `${summary.churn_rate}%`
              : "—"
          }
          accent="violet"
          subtitle="Observed churn in dataset"
        />

        <StatCard
          icon={<Target size={19} />}
          label="Churned Customers"
          value={
            summary
              ? summary.churned_customers.toLocaleString()
              : "—"
          }
          accent="rose"
          subtitle="Customers marked as churned"
        />

        <StatCard
          icon={<Brain size={19} />}
          label="Active Model"
          value={
            modelStatus?.active_model || "—"
          }
          accent="emerald"
          subtitle={
            modelStatus?.tabpfn_available
              ? "Primary AI model online"
              : "Backup model active"
          }
        />

      </div>


      {/* WORKFLOW + DISTRIBUTION */}

      <div className="grid gap-6 xl:grid-cols-[1.45fr_0.75fr]">

        <Card className="overflow-hidden">

          <div className="mb-6 flex items-start justify-between">

            <div>

              <div className="flex items-center gap-2 text-xs uppercase tracking-[0.2em] text-blue-400">
                <Zap size={13} />
                Intelligence Pipeline
              </div>

              <h2 className="mt-2 text-xl font-semibold">
                From customer data to retention action
              </h2>

              <p className="mt-2 max-w-xl text-sm leading-6 text-slate-500">
                Every prediction moves through the same structured
                pipeline before becoming a business recommendation.
              </p>

            </div>

            <div className="hidden rounded-xl border border-blue-400/10 bg-blue-400/5 p-3 sm:block">
              <Network
                size={21}
                className="text-blue-400"
              />
            </div>

          </div>

          <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-4">

            {[
              {
                number: "01",
                title: "Customer Data",
                text: "Profile, account and service attributes",
                icon: <Users size={16} />,
              },
              {
                number: "02",
                title: "TabPFN",
                text: "Predicts individual churn probability",
                icon: <Brain size={16} />,
              },
              {
                number: "03",
                title: "SHAP",
                text: "Identifies the strongest risk drivers",
                icon: <Sparkles size={16} />,
              },
              {
                number: "04",
                title: "Action",
                text: "Generates a targeted retention response",
                icon: <Target size={16} />,
              },
            ].map((step) => (
              <div
                key={step.number}
                className="group rounded-xl border border-white/10 bg-white/[0.02] p-4 transition hover:border-blue-400/20 hover:bg-blue-400/[0.03]"
              >

                <div className="flex items-center justify-between">

                  <span className="text-xs font-bold text-blue-400">
                    {step.number}
                  </span>

                  <span className="text-slate-600 transition group-hover:text-blue-400">
                    {step.icon}
                  </span>

                </div>

                <div className="mt-4 font-semibold">
                  {step.title}
                </div>

                <div className="mt-1 text-xs leading-5 text-slate-500">
                  {step.text}
                </div>

              </div>
            ))}

          </div>

          <div className="mt-5 rounded-xl border border-blue-400/10 bg-blue-400/[0.025] px-4 py-3">

            <div className="flex items-center gap-2 text-xs text-slate-400">

              <ShieldCheck
                size={14}
                className="text-emerald-400"
              />

              <span>
                Primary model:
              </span>

              <span className="font-semibold text-blue-300">
                {modelStatus?.primary_model || "TabPFN"}
              </span>

              <span className="text-slate-600">
                •
              </span>

              <span>
                Backup:
              </span>

              <span className="font-semibold text-violet-300">
                {modelStatus?.backup_model || "XGBoost"}
              </span>

            </div>

          </div>

        </Card>


        <Card>

          <div className="mb-2">

            <p className="text-xs uppercase tracking-[0.2em] text-slate-500">
              Dataset Composition
            </p>

            <h2 className="mt-2 text-xl font-semibold">
              Churn distribution
            </h2>

          </div>

          <div className="relative h-56">

            <ResponsiveContainer
              width="100%"
              height="100%"
            >

              <PieChart>

                <Pie
                  data={churnData}
                  dataKey="value"
                  nameKey="name"
                  innerRadius={58}
                  outerRadius={82}
                  paddingAngle={4}
                  stroke="none"
                >
                  {churnData.map((_, index) => (
                    <Cell
                      key={index}
                      fill={
                        index === 0
                          ? "#334155"
                          : "#8b5cf6"
                      }
                    />
                  ))}
                </Pie>

                <Tooltip
                  contentStyle={{
                    background: "#0b1020",
                    border: "1px solid rgba(255,255,255,0.1)",
                    borderRadius: 12,
                    color: "#fff",
                  }}
                  formatter={(value) =>
                    Number(value).toLocaleString()
                  }
                />

              </PieChart>

            </ResponsiveContainer>

            <div className="pointer-events-none absolute inset-0 flex items-center justify-center">

              <div className="text-center">

                <div className="text-2xl font-bold">
                  {summary
                    ? `${churnPercentage}%`
                    : "—"}
                </div>

                <div className="text-[10px] uppercase tracking-wider text-slate-500">
                  churn rate
                </div>

              </div>

            </div>

          </div>

          <div className="mt-2 grid grid-cols-2 gap-3">

            <div className="rounded-xl border border-white/10 bg-white/[0.02] p-3">

              <div className="flex items-center gap-2 text-xs text-slate-500">
                <span className="h-2 w-2 rounded-full bg-slate-600" />
                Retained
              </div>

              <div className="mt-1 text-lg font-semibold">
                {summary
                  ? summary.retained_customers.toLocaleString()
                  : "—"}
              </div>

            </div>

            <div className="rounded-xl border border-violet-400/10 bg-violet-400/[0.03] p-3">

              <div className="flex items-center gap-2 text-xs text-slate-500">
                <span className="h-2 w-2 rounded-full bg-violet-400" />
                Churned
              </div>

              <div className="mt-1 text-lg font-semibold text-violet-200">
                {summary
                  ? summary.churned_customers.toLocaleString()
                  : "—"}
              </div>

            </div>

          </div>

        </Card>

      </div>


      {/* ACTION CARDS */}

      <div>

        <div className="mb-4">

          <p className="text-xs uppercase tracking-[0.2em] text-slate-500">
            Quick Actions
          </p>

          <h2 className="mt-1 text-lg font-semibold">
            Explore the intelligence platform
          </h2>

        </div>

        <div className="grid gap-4 md:grid-cols-2">

          <button
            onClick={() => navigate("predict")}
            className="group relative overflow-hidden rounded-2xl border border-blue-400/20 bg-gradient-to-br from-blue-500/10 via-blue-500/[0.04] to-violet-500/[0.03] p-6 text-left transition duration-300 hover:-translate-y-0.5 hover:border-blue-400/40 hover:shadow-xl hover:shadow-blue-500/5"
          >

            <div className="absolute -right-10 -top-10 h-32 w-32 rounded-full bg-blue-500/10 blur-2xl transition group-hover:bg-blue-500/20" />

            <div className="relative">

              <div className="flex items-center justify-between">

                <div className="flex h-11 w-11 items-center justify-center rounded-xl bg-blue-500/10 text-blue-400">
                  <Target size={23} />
                </div>

                <ArrowUpRight
                  size={19}
                  className="text-slate-600 transition group-hover:-translate-y-0.5 group-hover:translate-x-0.5 group-hover:text-blue-400"
                />

              </div>

              <h3 className="mt-5 text-lg font-semibold">
                Predict Customer Churn
              </h3>

              <p className="mt-2 max-w-md text-sm leading-6 text-slate-400">
                Enter customer details and generate a live churn
                probability using the active ML model.
              </p>

              <div className="mt-5 text-sm font-medium text-blue-300">
                Open prediction engine →
              </div>

            </div>

          </button>


          <button
            onClick={() => navigate("dataset")}
            className="group relative overflow-hidden rounded-2xl border border-violet-400/15 bg-gradient-to-br from-violet-500/[0.07] to-transparent p-6 text-left transition duration-300 hover:-translate-y-0.5 hover:border-violet-400/30 hover:shadow-xl hover:shadow-violet-500/5"
          >

            <div className="absolute -right-10 -top-10 h-32 w-32 rounded-full bg-violet-500/10 blur-2xl transition group-hover:bg-violet-500/20" />

            <div className="relative">

              <div className="flex items-center justify-between">

                <div className="flex h-11 w-11 items-center justify-center rounded-xl bg-violet-500/10 text-violet-400">
                  <Database size={23} />
                </div>

                <ArrowUpRight
                  size={19}
                  className="text-slate-600 transition group-hover:-translate-y-0.5 group-hover:translate-x-0.5 group-hover:text-violet-400"
                />

              </div>

              <h3 className="mt-5 text-lg font-semibold">
                Explore Full Dataset
              </h3>

              <p className="mt-2 max-w-md text-sm leading-6 text-slate-400">
                Explore the original telecom churn dataset,
                including customer records and all source columns.
              </p>

              <div className="mt-5 text-sm font-medium text-violet-300">
                Open dataset explorer →
              </div>

            </div>

          </button>

        </div>

      </div>

    </div>
  );
}


/* ============================================================
   PREDICT PAGE
============================================================ */

function PredictPage({
  customer,
  updateCustomer,
  prediction,
  predicting,
  runPrediction,
}: {
  customer: typeof initialCustomer;
  updateCustomer: (
    key: keyof typeof initialCustomer,
    value: string | number
  ) => void;
  prediction: PredictionResult | null;
  predicting: boolean;
  runPrediction: () => void;
}) {
  return (
    <div className="space-y-8">

      <PageHeader
        eyebrow="Prediction Engine"
        title="Predict Customer Churn"
        description="Enter a customer's account and service profile to estimate churn probability and receive a retention recommendation."
      />

      <div className="grid gap-6 xl:grid-cols-[1.2fr_0.8fr]">

        <Card>

          <div className="grid gap-6 md:grid-cols-2">

            <FormSection
              title="Customer Profile"
              icon={<Users size={17} />}
            >

              <SelectField
                label="Gender"
                value={customer.gender}
                onChange={(value) =>
                  updateCustomer("gender", value)
                }
                options={["Male", "Female"]}
              />

              <BinaryField
                label="Senior Citizen"
                value={customer.senior_citizen}
                onChange={(value) =>
                  updateCustomer("senior_citizen", value)
                }
              />

              <BinaryField
                label="Partner"
                value={customer.partner}
                onChange={(value) =>
                  updateCustomer("partner", value)
                }
              />

              <BinaryField
                label="Dependents"
                value={customer.dependents}
                onChange={(value) =>
                  updateCustomer("dependents", value)
                }
              />

            </FormSection>

            <FormSection
              title="Account"
              icon={<Gauge size={17} />}
            >

              <NumberField
                label="Tenure (months)"
                value={customer.tenure}
                onChange={(value) =>
                  updateCustomer("tenure", value)
                }
              />

              <SelectField
                label="Contract"
                value={customer.contract}
                onChange={(value) =>
                  updateCustomer("contract", value)
                }
                options={[
                  "Month-to-month",
                  "One year",
                  "Two year",
                ]}
              />

              <SelectField
                label="Payment Method"
                value={customer.payment_method}
                onChange={(value) =>
                  updateCustomer("payment_method", value)
                }
                options={[
                  "Electronic check",
                  "Mailed check",
                  "Bank transfer (automatic)",
                  "Credit card (automatic)",
                ]}
              />

              <BinaryField
                label="Paperless Billing"
                value={customer.paperless_billing}
                onChange={(value) =>
                  updateCustomer("paperless_billing", value)
                }
              />

              <NumberField
                label="Monthly Charges"
                value={customer.monthly_charges}
                onChange={(value) =>
                  updateCustomer("monthly_charges", value)
                }
              />

              <NumberField
                label="Total Charges"
                value={customer.total_charges}
                onChange={(value) =>
                  updateCustomer("total_charges", value)
                }
              />

            </FormSection>

            <FormSection
              title="Services"
              icon={<Network size={17} />}
            >

              <BinaryField
                label="Phone Service"
                value={customer.phone_service}
                onChange={(value) =>
                  updateCustomer("phone_service", value)
                }
              />

              <BinaryField
                label="Multiple Lines"
                value={customer.multiple_lines}
                onChange={(value) =>
                  updateCustomer("multiple_lines", value)
                }
              />

              <SelectField
                label="Internet Service"
                value={customer.internet_service}
                onChange={(value) =>
                  updateCustomer("internet_service", value)
                }
                options={[
                  "DSL",
                  "Fiber optic",
                  "No",
                ]}
              />

              <BinaryField
                label="Online Security"
                value={customer.online_security}
                onChange={(value) =>
                  updateCustomer("online_security", value)
                }
              />

              <BinaryField
                label="Online Backup"
                value={customer.online_backup}
                onChange={(value) =>
                  updateCustomer("online_backup", value)
                }
              />

              <BinaryField
                label="Device Protection"
                value={customer.device_protection}
                onChange={(value) =>
                  updateCustomer("device_protection", value)
                }
              />

              <BinaryField
                label="Tech Support"
                value={customer.tech_support}
                onChange={(value) =>
                  updateCustomer("tech_support", value)
                }
              />

              <BinaryField
                label="Streaming TV"
                value={customer.streaming_tv}
                onChange={(value) =>
                  updateCustomer("streaming_tv", value)
                }
              />

              <BinaryField
                label="Streaming Movies"
                value={customer.streaming_movies}
                onChange={(value) =>
                  updateCustomer("streaming_movies", value)
                }
              />

            </FormSection>

          </div>

          <button
            onClick={runPrediction}
            disabled={predicting}
            className="mt-8 flex w-full items-center justify-center gap-2 rounded-xl bg-gradient-to-r from-blue-500 to-violet-600 px-5 py-4 font-semibold shadow-lg shadow-blue-500/20 transition hover:brightness-110 disabled:cursor-not-allowed disabled:opacity-60"
          >

            {predicting ? (
              <>
                <Activity size={18} className="animate-spin" />
                Running model...
              </>
            ) : (
              <>
                <Sparkles size={18} />
                Predict Churn
              </>
            )}

          </button>

        </Card>

        <PredictionCard prediction={prediction} />

      </div>

    </div>
  );
}


/* ============================================================
   PREDICTION RESULT
============================================================ */

function PredictionCard({
  prediction,
}: {
  prediction: PredictionResult | null;
}) {
  if (!prediction) {
    return (
      <Card className="flex min-h-[500px] items-center justify-center">

        <div className="max-w-sm text-center">

          <div className="mx-auto flex h-16 w-16 items-center justify-center rounded-2xl bg-blue-500/10 text-blue-400">
            <Brain size={30} />
          </div>

          <h2 className="mt-5 text-xl font-semibold">
            Ready for prediction
          </h2>

          <p className="mt-3 text-sm leading-6 text-slate-500">
            Complete the customer profile and run the model to see
            the churn probability, risk level, and recommended
            retention action.
          </p>

        </div>

      </Card>
    );
  }

  return (
    <Card>

      <div className="flex items-center justify-between">

        <div>

          <p className="text-xs uppercase tracking-[0.2em] text-slate-500">
            Prediction Result
          </p>

          <h2 className="mt-1 text-lg font-semibold">
            Churn Risk Assessment
          </h2>

        </div>

        <div className="rounded-lg border border-blue-400/20 bg-blue-400/5 px-3 py-1.5 text-xs text-blue-300">
          {prediction.model_used}
        </div>

      </div>

      <div className="mt-10 text-center">

        <div className="text-6xl font-bold tracking-tight">
          {prediction.churn_percentage}%
        </div>

        <div className="mt-2 text-sm text-slate-500">
          estimated churn probability
        </div>

        <div
          className={`
            mx-auto mt-5 inline-flex rounded-full px-4 py-2 text-sm font-semibold
            ${
              prediction.risk_level === "High"
                ? "bg-rose-500/10 text-rose-300"
                : prediction.risk_level === "Medium"
                ? "bg-amber-500/10 text-amber-300"
                : "bg-emerald-500/10 text-emerald-300"
            }
          `}
        >
          {prediction.risk_level} Risk
        </div>

      </div>

      <div className="mt-10 rounded-xl border border-white/10 bg-white/[0.025] p-5">

        <div className="flex items-center gap-2 text-sm font-semibold">

          <Sparkles
            size={17}
            className="text-violet-400"
          />

          Recommended Action

        </div>

        <p className="mt-3 text-sm leading-6 text-slate-400">
          {prediction.recommendation}
        </p>

      </div>

    </Card>
  );
}


/* ============================================================
   CUSTOMER EXPLANATION — REAL SHAP
============================================================ */

function ExplanationPage({
  prediction,
  explanation,
  explaining,
  navigate,
}: {
  prediction: PredictionResult | null;
  explanation: ExplanationResult | null;
  explaining: boolean;
  navigate: (page: Page) => void;
}) {
  return (
    <div className="space-y-8">

      <PageHeader
        eyebrow="Customer Intelligence"
        title="Customer Explanation"
        description="Understand why the model considers this customer likely or unlikely to churn using real SHAP feature attribution."
      />

      {!prediction ? (
        <Card className="py-20 text-center">

          <Sparkles
            className="mx-auto text-violet-400"
            size={35}
          />

          <h2 className="mt-5 text-xl font-semibold">
            No prediction available
          </h2>

          <p className="mx-auto mt-3 max-w-md text-sm leading-6 text-slate-500">
            Run a customer prediction first. The actual TabPFN
            prediction and SHAP explanation will appear here.
          </p>

          <button
            onClick={() => navigate("predict")}
            className="mt-6 rounded-xl bg-blue-500 px-5 py-3 text-sm font-semibold"
          >
            Open Predictor
          </button>

        </Card>
      ) : (
        <>
          <div className="grid gap-4 md:grid-cols-3">

            <InfoCard
              title="Risk Level"
              value={`${prediction.risk_level} Risk`}
              icon={<ShieldCheck size={19} />}
            />

            <InfoCard
              title="Churn Probability"
              value={`${prediction.churn_percentage}%`}
              icon={<Target size={19} />}
            />

            <InfoCard
              title="Model Used"
              value={prediction.model_used}
              icon={<Brain size={19} />}
            />

          </div>

          <Card>

            <div className="flex flex-col gap-4 sm:flex-row sm:items-center sm:justify-between">

              <div>

                <p className="text-xs uppercase tracking-[0.2em] text-slate-500">
                  Explainable AI
                </p>

                <h2 className="mt-1 text-xl font-semibold">
                  Why this customer is at risk
                </h2>

              </div>

              <div className="w-fit rounded-full border border-violet-400/20 bg-violet-400/5 px-3 py-1.5 text-xs text-violet-300">
                {explanation?.explanation_method || "SHAP"}
              </div>

            </div>

            {explaining ? (
              <div className="flex min-h-[280px] items-center justify-center">

                <div className="text-center">

                  <Activity
                    size={30}
                    className="mx-auto animate-spin text-violet-400"
                  />

                  <p className="mt-4 text-sm font-medium text-slate-300">
                    Calculating SHAP explanation...
                  </p>

                  <p className="mt-2 text-xs text-slate-500">
                    TabPFN is analyzing the customer's individual
                    feature contributions.
                  </p>

                </div>

              </div>
            ) : explanation ? (
              <>
                <div className="mt-6 rounded-xl border border-blue-400/10 bg-blue-400/[0.03] p-4">

                  <div className="flex items-center gap-2">

                    <Sparkles
                      size={17}
                      className="text-blue-400"
                    />

                    <span className="text-sm font-semibold">
                      Individual prediction explanation
                    </span>

                  </div>

                  <p className="mt-2 text-xs leading-5 text-slate-500">
                    Positive SHAP values increase predicted churn
                    risk, while negative values decrease it.
                  </p>

                </div>

                <div className="mt-6">

                  <div className="mb-4 flex items-center justify-between">

                    <h3 className="text-sm font-semibold text-slate-300">
                      Top contributing factors
                    </h3>

                    <span className="text-xs text-slate-600">
                      Real model attribution
                    </span>

                  </div>

                  <div className="space-y-3">

                    {explanation.top_factors.map(
                      (factor, index) => {

                        const barWidth = Math.min(
                          100,
                          Math.max(
                            8,
                            factor.absolute_impact * 1000
                          )
                        );

                        return (
                          <div
                            key={`${factor.feature}-${index}`}
                            className="rounded-xl border border-white/10 bg-white/[0.02] p-4"
                          >

                            <div className="flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">

                              <div className="min-w-0">

                                <div className="font-semibold text-slate-200">
                                  {factor.name}
                                </div>

                                <div className="mt-1 text-xs text-slate-500">
                                  Feature value: {factor.value}
                                </div>

                              </div>

                              <div
                                className={`w-fit shrink-0 rounded-full px-3 py-1.5 text-xs font-semibold ${
                                  factor.direction === "increases"
                                    ? "bg-rose-500/10 text-rose-300"
                                    : "bg-emerald-500/10 text-emerald-300"
                                }`}
                              >
                                {factor.direction === "increases"
                                  ? "↑ Higher churn risk"
                                  : "↓ Lower churn risk"}
                              </div>

                            </div>

                            <div className="mt-4">

                              <div className="flex items-center justify-between text-xs">

                                <span className="text-slate-500">
                                  SHAP contribution
                                </span>

                                <span
                                  className={
                                    factor.shap_value >= 0
                                      ? "font-semibold text-rose-300"
                                      : "font-semibold text-emerald-300"
                                  }
                                >
                                  {factor.shap_value >= 0
                                    ? "+"
                                    : ""}
                                  {factor.shap_value.toFixed(4)}
                                </span>

                              </div>

                              <div className="mt-2 h-2 overflow-hidden rounded-full bg-white/5">

                                <div
                                  className={`h-full rounded-full ${
                                    factor.shap_value >= 0
                                      ? "bg-rose-400"
                                      : "bg-emerald-400"
                                  }`}
                                  style={{
                                    width: `${barWidth}%`,
                                  }}
                                />

                              </div>

                            </div>

                          </div>
                        );
                      }
                    )}

                  </div>

                </div>

                {explanation.negative_factors.length > 0 && (
                  <div className="mt-7">

                    <h3 className="mb-3 text-sm font-semibold text-emerald-300">
                      Factors reducing churn risk
                    </h3>

                    <div className="space-y-2">

                      {explanation.negative_factors.map(
                        (factor, index) => (
                          <div
                            key={`${factor.feature}-negative-${index}`}
                            className="flex items-center justify-between rounded-xl border border-emerald-400/10 bg-emerald-400/[0.03] p-4"
                          >

                            <span className="text-sm text-slate-300">
                              {factor.name}
                            </span>

                            <span className="text-xs font-semibold text-emerald-300">
                              {factor.shap_value.toFixed(4)}
                            </span>

                          </div>
                        )
                      )}

                    </div>

                  </div>
                )}

                <div className="mt-7 rounded-xl border border-white/10 bg-white/[0.02] p-4">

                  <div className="flex items-center gap-2 text-sm font-semibold">

                    <Info
                      size={16}
                      className="text-slate-400"
                    />

                    Interpretation

                  </div>

                  <p className="mt-2 text-xs leading-6 text-slate-500">
                    {explanation.note}
                  </p>

                  <p className="mt-2 text-[11px] text-slate-600">
                    SHAP explanation calculated using{" "}
                    {explanation.background_samples} representative
                    background samples.
                  </p>

                </div>
              </>
            ) : (
              <div className="py-16 text-center">

                <Info
                  size={28}
                  className="mx-auto text-amber-400"
                />

                <p className="mt-4 text-sm text-slate-400">
                  The prediction is available, but the SHAP
                  explanation could not be loaded.
                </p>

                <p className="mt-2 text-xs text-slate-600">
                  Check that the FastAPI backend is running.
                </p>

              </div>
            )}

          </Card>

          <Card>

            <h2 className="text-lg font-semibold">
              Retention Recommendation
            </h2>

            <p className="mt-3 max-w-3xl text-sm leading-7 text-slate-400">
              {prediction.recommendation}
            </p>

          </Card>
        </>
      )}

    </div>
  );
}


/* ============================================================
   ANALYTICS
============================================================ */

function AnalyticsPage({
  analytics,
}: {
  analytics: AnalyticsData | null;
}) {
  const contractData = analytics
    ? Object.entries(analytics.contract_churn).map(
        ([name, value]) => ({
          name,
          value,
        })
      )
    : [];

  const internetData = analytics
    ? Object.entries(analytics.internet_service_churn).map(
        ([name, value]) => ({
          name,
          value,
        })
      )
    : [];

  const paymentData = analytics
    ? Object.entries(analytics.payment_method_churn).map(
        ([name, value]) => ({
          name,
          value,
        })
      )
    : [];

  return (
    <div className="space-y-8">

      <PageHeader
        eyebrow="Behavioral Analytics"
        title="Churn Analytics"
        description="Explore how churn varies across customer contract, internet service, payment method, and demographic segments."
      />

      <div className="grid gap-6 xl:grid-cols-2">

        <ChartCard
          title="Churn by Contract"
          data={contractData}
        />

        <ChartCard
          title="Churn by Internet Service"
          data={internetData}
        />

        <ChartCard
          title="Churn by Payment Method"
          data={paymentData}
        />

        <Card>

          <div className="mb-5">

            <p className="text-xs uppercase tracking-[0.2em] text-slate-500">
              Demographic Signal
            </p>

            <h2 className="mt-1 text-lg font-semibold">
              Senior Citizen Churn
            </h2>

          </div>

          <div className="h-72">

            <ResponsiveContainer
              width="100%"
              height="100%"
            >

              <BarChart
                data={
                  analytics
                    ? Object.entries(
                        analytics.senior_citizen_churn
                      ).map(([name, value]) => ({
                        name:
                          name === "1"
                            ? "Senior Citizen"
                            : "Non-Senior",
                        value,
                      }))
                    : []
                }
              >

                <CartesianGrid
                  strokeDasharray="3 3"
                  stroke="rgba(255,255,255,0.06)"
                />

                <XAxis
                  dataKey="name"
                  stroke="#64748b"
                  fontSize={11}
                />

                <YAxis
                  stroke="#64748b"
                  fontSize={11}
                />

                <Tooltip
                  contentStyle={{
                    background: "#0b1020",
                    border: "1px solid rgba(255,255,255,0.1)",
                    borderRadius: 12,
                  }}
                />

                <Bar
                  dataKey="value"
                  fill="#6366f1"
                  radius={[8, 8, 0, 0]}
                />

              </BarChart>

            </ResponsiveContainer>

          </div>

        </Card>

      </div>

    </div>
  );
}


/* ============================================================
   MODEL PERFORMANCE
============================================================ */

function PerformancePage({
  results,
  loading,
  modelStatus,
}: {
  results: ModelResult[];
  loading: boolean;
  modelStatus: ModelStatus | null;
}) {
  const primary =
    results.find(
      (model) =>
        model.name === modelStatus?.primary_model
    );

  const metricChartData = results
    .filter(
      (model) =>
        model.accuracy !== undefined
    )
    .map((model) => ({
      name: model.name,
      Accuracy: model.accuracy,
      Precision: model.precision,
      Recall: model.recall,
      F1: model.f1_score,
      "ROC-AUC": model.roc_auc,
    }));

  return (
    <div className="space-y-8">

      <PageHeader
        eyebrow="Model Evaluation"
        title="Model Performance"
        description="Real evaluation results calculated on the untouched 20% test set used by the project."
      />

      <div className="grid gap-4 md:grid-cols-3">

        <StatCard
          icon={<Brain size={19} />}
          label="Primary Model"
          value={
            modelStatus?.primary_model || "—"
          }
          accent="blue"
          subtitle="Designated primary classifier"
        />

        <StatCard
          icon={<ShieldCheck size={19} />}
          label="Backup Model"
          value={
            modelStatus?.backup_model || "—"
          }
          accent="violet"
          subtitle="Fallback classifier"
        />

        <StatCard
          icon={<Database size={19} />}
          label="Test Samples"
          value="1,409"
          accent="emerald"
          subtitle="Held-out evaluation records"
        />

      </div>

      <Card>

        <div className="mb-6 flex items-start justify-between">

          <div>

            <p className="text-xs uppercase tracking-[0.2em] text-slate-500">
              Evaluation Metrics
            </p>

            <h2 className="mt-1 text-lg font-semibold">
              Actual Test-Set Performance
            </h2>

          </div>

          {primary && (
            <div className="rounded-full border border-blue-400/20 bg-blue-400/5 px-3 py-1.5 text-xs text-blue-300">
              Primary: {primary.name}
            </div>
          )}

        </div>

        {loading ? (
          <div className="flex h-60 items-center justify-center text-sm text-slate-500">
            Evaluating models...
          </div>
        ) : (
          <div className="overflow-x-auto">

            <table className="w-full min-w-[900px] text-left">

              <thead>

                <tr className="border-b border-white/10 text-xs uppercase tracking-wider text-slate-500">

                  <th className="px-4 py-4">Model</th>
                  <th className="px-4 py-4">Role</th>
                  <th className="px-4 py-4">Accuracy</th>
                  <th className="px-4 py-4">Precision</th>
                  <th className="px-4 py-4">Recall</th>
                  <th className="px-4 py-4">F1</th>
                  <th className="px-4 py-4">ROC-AUC</th>
                  <th className="px-4 py-4">PR-AUC</th>

                </tr>

              </thead>

              <tbody>

                {results.map((model) => (

                  <tr
                    key={model.name}
                    className="border-b border-white/5 text-sm"
                  >

                    <td className="px-4 py-4 font-semibold">
                      {model.name}
                    </td>

                    <td className="px-4 py-4">

                      <span
                        className={`
                          rounded-full px-2.5 py-1 text-xs
                          ${
                            model.role === "Primary"
                              ? "bg-blue-500/10 text-blue-300"
                              : model.role === "Backup"
                              ? "bg-violet-500/10 text-violet-300"
                              : "bg-white/5 text-slate-400"
                          }
                        `}
                      >
                        {model.role}
                      </span>

                    </td>

                    <td className="px-4 py-4 text-slate-300">
                      {formatMetric(model.accuracy)}
                    </td>

                    <td className="px-4 py-4 text-slate-300">
                      {formatMetric(model.precision)}
                    </td>

                    <td className="px-4 py-4 text-slate-300">
                      {formatMetric(model.recall)}
                    </td>

                    <td className="px-4 py-4 text-slate-300">
                      {formatMetric(model.f1_score)}
                    </td>

                    <td className="px-4 py-4 text-blue-300">
                      {formatMetric(model.roc_auc)}
                    </td>

                    <td className="px-4 py-4 text-violet-300">
                      {formatMetric(model.pr_auc)}
                    </td>

                  </tr>

                ))}

              </tbody>

            </table>

          </div>
        )}

      </Card>

      <Card>

        <div className="mb-6">

          <p className="text-xs uppercase tracking-[0.2em] text-slate-500">
            Comparison
          </p>

          <h2 className="mt-1 text-lg font-semibold">
            Model Metric Comparison
          </h2>

        </div>

        <div className="h-[380px]">

          <ResponsiveContainer
            width="100%"
            height="100%"
          >

            <BarChart
              data={metricChartData}
              margin={{
                top: 10,
                right: 20,
                left: 0,
                bottom: 10,
              }}
            >

              <CartesianGrid
                strokeDasharray="3 3"
                stroke="rgba(255,255,255,0.06)"
              />

              <XAxis
                dataKey="name"
                stroke="#64748b"
                fontSize={11}
              />

              <YAxis
                domain={[0, 100]}
                stroke="#64748b"
                fontSize={11}
              />

              <Tooltip
                formatter={(value) =>
                  `${Number(value).toFixed(2)}%`
                }
                contentStyle={{
                  background: "#0b1020",
                  border: "1px solid rgba(255,255,255,0.1)",
                  borderRadius: 12,
                }}
              />

              <Legend />

              <Bar
                dataKey="Accuracy"
                fill="#3b82f6"
                radius={[5, 5, 0, 0]}
              />

              <Bar
                dataKey="Precision"
                fill="#8b5cf6"
                radius={[5, 5, 0, 0]}
              />

              <Bar
                dataKey="Recall"
                fill="#06b6d4"
                radius={[5, 5, 0, 0]}
              />

              <Bar
                dataKey="F1"
                fill="#10b981"
                radius={[5, 5, 0, 0]}
              />

              <Bar
                dataKey="ROC-AUC"
                fill="#f59e0b"
                radius={[5, 5, 0, 0]}
              />

            </BarChart>

          </ResponsiveContainer>

        </div>

      </Card>

      <div className="grid gap-6 md:grid-cols-2">

        {results.map((model) => (

          <Card key={model.name}>

            <div className="flex items-center justify-between">

              <div>

                <h3 className="font-semibold">
                  {model.name}
                </h3>

                <p className="mt-1 text-xs text-slate-500">
                  {model.role} model
                </p>

              </div>

              <div
                className={`
                  rounded-full px-3 py-1 text-xs
                  ${
                    model.status === "Available"
                      ? "bg-emerald-500/10 text-emerald-300"
                      : "bg-rose-500/10 text-rose-300"
                  }
                `}
              >
                {model.status}
              </div>

            </div>

            {model.confusion_matrix && (

              <div className="mt-6 grid grid-cols-2 gap-2 text-center text-xs">

                <div className="rounded-lg bg-emerald-500/5 p-3">
                  <div className="text-lg font-semibold text-emerald-300">
                    {model.confusion_matrix.true_negative}
                  </div>
                  <div className="text-slate-500">
                    True Negative
                  </div>
                </div>

                <div className="rounded-lg bg-rose-500/5 p-3">
                  <div className="text-lg font-semibold text-rose-300">
                    {model.confusion_matrix.false_positive}
                  </div>
                  <div className="text-slate-500">
                    False Positive
                  </div>
                </div>

                <div className="rounded-lg bg-amber-500/5 p-3">
                  <div className="text-lg font-semibold text-amber-300">
                    {model.confusion_matrix.false_negative}
                  </div>
                  <div className="text-slate-500">
                    False Negative
                  </div>
                </div>

                <div className="rounded-lg bg-blue-500/5 p-3">
                  <div className="text-lg font-semibold text-blue-300">
                    {model.confusion_matrix.true_positive}
                  </div>
                  <div className="text-slate-500">
                    True Positive
                  </div>
                </div>

              </div>

            )}

          </Card>

        ))}

      </div>

      <div className="rounded-xl border border-amber-400/10 bg-amber-400/[0.03] p-5">

        <div className="flex gap-3">

          <Info
            size={18}
            className="mt-0.5 shrink-0 text-amber-400"
          />

          <p className="text-sm leading-6 text-slate-400">
            These metrics are calculated dynamically from the held-out
            test set. TabPFN remains the designated primary model and
            XGBoost the backup model; the website does not claim TabPFN
            is the highest-accuracy model unless the measured results
            support that claim.
          </p>

        </div>

      </div>

    </div>
  );
}


/* ============================================================
   EXPLAINABLE AI
============================================================ */

function XAIPage() {
  return (
    <div className="space-y-8">

      <PageHeader
        eyebrow="Interpretability"
        title="Explainable AI"
        description="Understand why a customer receives a churn prediction using SHAP-based model interpretation."
      />

      <div className="grid gap-6 md:grid-cols-3">

        <InfoCard
          title="What is SHAP?"
          value="Feature attribution"
          icon={<Sparkles size={19} />}
        />

        <InfoCard
          title="Primary Model"
          value="TabPFN"
          icon={<Brain size={19} />}
        />

        <InfoCard
          title="Goal"
          value="Understand why"
          icon={<Target size={19} />}
        />

      </div>

      <Card>

        <div className="grid gap-8 lg:grid-cols-2">

          <div>

            <p className="text-xs uppercase tracking-[0.2em] text-slate-500">
              Explanation Method
            </p>

            <h2 className="mt-2 text-2xl font-semibold">
              SHAP connects predictions to customer behavior
            </h2>

            <p className="mt-4 text-sm leading-7 text-slate-400">
              SHAP assigns contribution values to input features,
              helping us understand which customer characteristics
              push a prediction toward higher or lower churn risk.
            </p>

          </div>

          <div className="space-y-3">

            {[
              [
                "Contract",
                "Contract type can strongly influence retention behavior.",
              ],
              [
                "Tenure",
                "Shorter customer relationships can indicate early-life churn risk.",
              ],
              [
                "Monthly Charges",
                "Higher recurring charges may contribute to customer risk.",
              ],
              [
                "Internet Service",
                "Service type can reveal differences in churn behavior.",
              ],
            ].map(([title, text]) => (

              <div
                key={title}
                className="rounded-xl border border-white/10 bg-white/[0.02] p-4"
              >

                <div className="font-semibold">
                  {title}
                </div>

                <div className="mt-1 text-xs leading-5 text-slate-500">
                  {text}
                </div>

              </div>

            ))}

          </div>

        </div>

      </Card>

      <div className="rounded-xl border border-blue-400/10 bg-blue-400/[0.03] p-5">

        <div className="flex gap-3">

          <Info
            size={18}
            className="mt-0.5 shrink-0 text-blue-400"
          />

          <p className="text-sm leading-6 text-slate-400">
            The project's notebook uses SHAP to explain TabPFN
            predictions and also performs an XGBoost TreeSHAP cross-check.
            Individual customer explanations shown in Customer Explanation
            are generated live by the FastAPI backend using the actual
            model prediction.
          </p>

        </div>

      </div>

    </div>
  );
}


/* ============================================================
   DATASET
============================================================ */

function DatasetPage({
  dataset,
  loading,
  page,
  setPage,
  search,
  setSearch,
}: {
  dataset: DatasetResponse | null;
  loading: boolean;
  page: number;
  setPage: Dispatch<SetStateAction<number>>;
  search: string;
  setSearch: (value: string) => void;
}) {
  return (
    <div className="space-y-8">

      <PageHeader
        eyebrow="Data Explorer"
        title="Full Telecom Churn Dataset"
        description="Explore the original customer-level dataset used by the project. Search across all source columns and navigate through the complete records."
      />

      <Card>

        <div className="flex flex-col gap-4 md:flex-row md:items-center md:justify-between">

          <div className="relative w-full md:max-w-md">

            <Search
              size={17}
              className="absolute left-3 top-1/2 -translate-y-1/2 text-slate-500"
            />

            <input
              value={search}
              onChange={(event) =>
                setSearch(event.target.value)
              }
              placeholder="Search dataset..."
              className="w-full rounded-xl border border-white/10 bg-white/[0.03] py-3 pl-10 pr-4 text-sm outline-none transition focus:border-blue-400/40"
            />

          </div>

          <div className="text-xs text-slate-500">

            {dataset
              ? `${dataset.total_rows.toLocaleString()} matching rows`
              : "Loading dataset..."}

          </div>

        </div>

        <div className="mt-6 overflow-x-auto rounded-xl border border-white/10">

          {loading ? (

            <div className="flex h-80 items-center justify-center text-sm text-slate-500">
              Loading dataset...
            </div>

          ) : dataset ? (

            <table className="w-full min-w-[1800px] text-left text-xs">

              <thead className="bg-white/[0.03]">

                <tr>

                  {dataset.columns.map((column) => (

                    <th
                      key={column}
                      className="whitespace-nowrap border-b border-white/10 px-4 py-3 font-semibold text-slate-400"
                    >
                      {column}
                    </th>

                  ))}

                </tr>

              </thead>

              <tbody>

                {dataset.data.map((row, rowIndex) => (

                  <tr
                    key={rowIndex}
                    className="border-b border-white/5 hover:bg-white/[0.025]"
                  >

                    {dataset.columns.map((column) => {

                      const value = row[column];

                      const isChurn =
                        column === "Churn" &&
                        value === "Yes";

                      return (
                        <td
                          key={column}
                          className={`
                            whitespace-nowrap px-4 py-3
                            ${
                              isChurn
                                ? "font-semibold text-rose-300"
                                : "text-slate-400"
                            }
                          `}
                        >
                          {String(value ?? "")}
                        </td>
                      );

                    })}

                  </tr>

                ))}

              </tbody>

            </table>

          ) : (

            <div className="flex h-80 items-center justify-center text-sm text-slate-500">
              Dataset unavailable.
            </div>

          )}

        </div>

        {dataset && (

          <div className="mt-5 flex items-center justify-between">

            <div className="text-xs text-slate-500">
              Page {dataset.page} of {dataset.total_pages}
            </div>

            <div className="flex items-center gap-2">

              <button
                disabled={page <= 1}
                onClick={() =>
                  setPage((current) =>
                    Math.max(1, current - 1)
                  )
                }
                className="rounded-lg border border-white/10 p-2 text-slate-400 transition hover:bg-white/5 disabled:cursor-not-allowed disabled:opacity-30"
              >
                <ChevronLeft size={17} />
              </button>

              <button
                disabled={
                  page >= dataset.total_pages
                }
                onClick={() =>
                  setPage((current) =>
                    Math.min(
                      dataset.total_pages,
                      current + 1
                    )
                  )
                }
                className="rounded-lg border border-white/10 p-2 text-slate-400 transition hover:bg-white/5 disabled:cursor-not-allowed disabled:opacity-30"
              >
                <ChevronRight size={17} />
              </button>

            </div>

          </div>

        )}

      </Card>

    </div>
  );
}


/* ============================================================
   ABOUT
============================================================ */

function AboutPage() {
  return (
    <div className="space-y-8">

      <PageHeader
        eyebrow="Project"
        title="About the Project"
        description="An end-to-end machine learning system for telecom customer churn prediction and explainable retention decision support."
      />

      <div className="grid gap-6 md:grid-cols-2">

        <Card>

          <div className="flex items-center gap-3">

            <div className="rounded-xl bg-blue-500/10 p-3 text-blue-400">
              <Brain size={22} />
            </div>

            <h2 className="text-lg font-semibold">
              Machine Learning
            </h2>

          </div>

          <p className="mt-5 text-sm leading-7 text-slate-400">
            The project evaluates Logistic Regression, Random Forest,
            XGBoost, and TabPFN. TabPFN is configured as the primary
            model, with XGBoost retained as the backup model.
          </p>

        </Card>

        <Card>

          <div className="flex items-center gap-3">

            <div className="rounded-xl bg-violet-500/10 p-3 text-violet-400">
              <Sparkles size={22} />
            </div>

            <h2 className="text-lg font-semibold">
              Explainability
            </h2>

          </div>

          <p className="mt-5 text-sm leading-7 text-slate-400">
            SHAP is used to interpret model predictions and connect
            churn risk to customer-level features and actionable
            retention strategies.
          </p>

        </Card>

        <Card>

          <div className="flex items-center gap-3">

            <div className="rounded-xl bg-emerald-500/10 p-3 text-emerald-400">
              <Network size={22} />
            </div>

            <h2 className="text-lg font-semibold">
              Technology Stack
            </h2>

          </div>

          <p className="mt-5 text-sm leading-7 text-slate-400">
            React, TypeScript, Tailwind CSS, Recharts, FastAPI,
            Python, pandas, scikit-learn, XGBoost, TabPFN, and SHAP.
          </p>

        </Card>

        <Card>

          <div className="flex items-center gap-3">

            <div className="rounded-xl bg-amber-500/10 p-3 text-amber-400">
              <Target size={22} />
            </div>

            <h2 className="text-lg font-semibold">
              Business Goal
            </h2>

          </div>

          <p className="mt-5 text-sm leading-7 text-slate-400">
            Move from simply predicting churn to understanding why
            customers are at risk and recommending targeted retention
            actions.
          </p>

        </Card>

      </div>

    </div>
  );
}


/* ============================================================
   REUSABLE COMPONENTS
============================================================ */

function PageHeader({
  eyebrow,
  title,
  description,
}: {
  eyebrow: string;
  title: string;
  description: string;
}) {
  return (
    <div>

      <p className="text-xs uppercase tracking-[0.25em] text-blue-400">
        {eyebrow}
      </p>

      <h1 className="mt-2 text-3xl font-bold tracking-tight sm:text-4xl">
        {title}
      </h1>

      <p className="mt-3 max-w-3xl text-sm leading-6 text-slate-500">
        {description}
      </p>

    </div>
  );
}


function Card({
  children,
  className = "",
}: {
  children: ReactNode;
  className?: string;
}) {
  return (
    <div
      className={`
        rounded-2xl
        border border-white/10
        bg-white/[0.025]
        p-5 sm:p-6
        ${className}
      `}
    >
      {children}
    </div>
  );
}


function StatCard({
  icon,
  label,
  value,
  accent,
  subtitle,
}: {
  icon: ReactNode;
  label: string;
  value: string;
  accent: "blue" | "violet" | "rose" | "emerald";
  subtitle: string;
}) {
  const accentStyles = {
    blue: {
      box: "bg-blue-500/10",
      icon: "text-blue-400",
      glow: "group-hover:border-blue-400/20",
    },
    violet: {
      box: "bg-violet-500/10",
      icon: "text-violet-400",
      glow: "group-hover:border-violet-400/20",
    },
    rose: {
      box: "bg-rose-500/10",
      icon: "text-rose-400",
      glow: "group-hover:border-rose-400/20",
    },
    emerald: {
      box: "bg-emerald-500/10",
      icon: "text-emerald-400",
      glow: "group-hover:border-emerald-400/20",
    },
  };

  const style = accentStyles[accent];

  return (
    <div
      className={`
        group rounded-2xl border border-white/10
        bg-white/[0.025] p-5 transition
        hover:-translate-y-0.5
        ${style.glow}
      `}
    >

      <div
        className={`inline-flex rounded-xl p-2.5 ${style.box} ${style.icon}`}
      >
        {icon}
      </div>

      <div className="mt-5 text-xs uppercase tracking-wider text-slate-500">
        {label}
      </div>

      <div className="mt-1 truncate text-2xl font-bold tracking-tight">
        {value}
      </div>

      <div className="mt-2 text-xs text-slate-600">
        {subtitle}
      </div>

    </div>
  );
}


function InfoCard({
  title,
  value,
  icon,
}: {
  title: string;
  value: string;
  icon: ReactNode;
}) {
  return (
    <Card>

      <div className="flex items-center gap-3 text-slate-400">

        <div className="rounded-lg bg-blue-500/10 p-2 text-blue-400">
          {icon}
        </div>

        <span className="text-xs uppercase tracking-wider">
          {title}
        </span>

      </div>

      <div className="mt-5 text-xl font-semibold">
        {value}
      </div>

    </Card>
  );
}


function FormSection({
  title,
  icon,
  children,
}: {
  title: string;
  icon: ReactNode;
  children: ReactNode;
}) {
  return (
    <div className="rounded-2xl border border-white/10 bg-white/[0.02] p-5">

      <div className="mb-5 flex items-center gap-2">

        <span className="text-blue-400">
          {icon}
        </span>

        <h3 className="font-semibold">
          {title}
        </h3>

      </div>

      <div className="space-y-4">
        {children}
      </div>

    </div>
  );
}


function SelectField({
  label,
  value,
  options,
  onChange,
}: {
  label: string;
  value: string;
  options: string[];
  onChange: (value: string) => void;
}) {
  return (
    <label className="block">

      <span className="mb-2 block text-xs text-slate-500">
        {label}
      </span>

      <select
        value={value}
        onChange={(event) =>
          onChange(event.target.value)
        }
        className="w-full rounded-xl border border-white/10 bg-[#0b1020] px-3 py-3 text-sm outline-none focus:border-blue-400/40"
      >

        {options.map((option) => (
          <option
            key={option}
            value={option}
          >
            {option}
          </option>
        ))}

      </select>

    </label>
  );
}


function NumberField({
  label,
  value,
  onChange,
}: {
  label: string;
  value: number;
  onChange: (value: number) => void;
}) {
  return (
    <label className="block">

      <span className="mb-2 block text-xs text-slate-500">
        {label}
      </span>

      <input
        type="number"
        value={value}
        onChange={(event) =>
          onChange(
            Number(event.target.value)
          )
        }
        className="w-full rounded-xl border border-white/10 bg-[#0b1020] px-3 py-3 text-sm outline-none focus:border-blue-400/40"
      />

    </label>
  );
}


function BinaryField({
  label,
  value,
  onChange,
}: {
  label: string;
  value: number;
  onChange: (value: number) => void;
}) {
  return (
    <label className="block">

      <span className="mb-2 block text-xs text-slate-500">
        {label}
      </span>

      <select
        value={value}
        onChange={(event) =>
          onChange(
            Number(event.target.value)
          )
        }
        className="w-full rounded-xl border border-white/10 bg-[#0b1020] px-3 py-3 text-sm outline-none focus:border-blue-400/40"
      >

        <option value={0}>
          No
        </option>

        <option value={1}>
          Yes
        </option>

      </select>

    </label>
  );
}


function ChartCard({
  title,
  data,
}: {
  title: string;
  data: {
    name: string;
    value: number;
  }[];
}) {
  return (
    <Card>

      <div className="mb-5">

        <p className="text-xs uppercase tracking-[0.2em] text-slate-500">
          Segment Analysis
        </p>

        <h2 className="mt-1 text-lg font-semibold">
          {title}
        </h2>

      </div>

      <div className="h-72">

        <ResponsiveContainer
          width="100%"
          height="100%"
        >

          <BarChart data={data}>

            <CartesianGrid
              strokeDasharray="3 3"
              stroke="rgba(255,255,255,0.06)"
            />

            <XAxis
              dataKey="name"
              stroke="#64748b"
              fontSize={10}
            />

            <YAxis
              stroke="#64748b"
              fontSize={11}
            />

            <Tooltip
              formatter={(value) =>
                `${Number(value).toFixed(2)}%`
              }
              contentStyle={{
                background: "#0b1020",
                border: "1px solid rgba(255,255,255,0.1)",
                borderRadius: 12,
              }}
            />

            <Line
              type="monotone"
              dataKey="value"
            />

            <Bar
              dataKey="value"
              fill="#6366f1"
              radius={[8, 8, 0, 0]}
            />

          </BarChart>

        </ResponsiveContainer>

      </div>

    </Card>
  );
}


function formatMetric(value?: number) {
  if (
    value === undefined ||
    value === null
  ) {
    return "—";
  }

  return `${value.toFixed(2)}%`;
}


export default App;