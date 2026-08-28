import React, { useMemo, useState } from "react";
import { createRoot } from "react-dom/client";
import {
  AlertTriangle,
  CheckCircle2,
  FileSearch,
  RefreshCcw,
  Search,
  ShieldCheck,
  XCircle,
} from "lucide-react";
import { checkPurchase, searchExistingPurchases } from "./api";
import "./styles.css";

const emptyForm = {
  product_name: "",
  fk_central_unit: "",
  order_id: "",
  supply_order_date: "",
};

function decisionLabel(decision) {
  if (decision === "duplicate") return "Duplicate";
  if (decision === "manual_review") return "Manual review";
  if (decision === "clear") return "Clear";
  return "Not checked";
}

function decisionIcon(decision) {
  if (decision === "duplicate") return <XCircle size={18} />;
  if (decision === "manual_review") return <AlertTriangle size={18} />;
  if (decision === "clear") return <CheckCircle2 size={18} />;
  return <ShieldCheck size={18} />;
}

function App() {
  const [form, setForm] = useState(emptyForm);
  const [search, setSearch] = useState({ query: "", fk_central_unit: "" });
  const [existingRows, setExistingRows] = useState([]);
  const [selectedRow, setSelectedRow] = useState(null);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [searching, setSearching] = useState(false);
  const [error, setError] = useState("");

  const canCheck = useMemo(
    () =>
      form.product_name.trim() &&
      String(form.fk_central_unit).trim() &&
      form.order_id.trim() &&
      form.supply_order_date,
    [form],
  );

  function updateForm(field, value) {
    setForm((current) => ({ ...current, [field]: value }));
    setResult(null);
    setError("");
  }

  function resetInput() {
    setForm(emptyForm);
    setSelectedRow(null);
    setResult(null);
    setError("");
  }

  function selectExisting(row) {
    setSelectedRow(row);
    setForm({
      product_name: row.product_name || "",
      fk_central_unit: row.fk_central_unit || "",
      order_id: row.order_id || "",
      supply_order_date: row.supply_order_date || "",
    });
    setResult(null);
    setError("");
  }

  async function handleSearchExisting(event) {
    event.preventDefault();
    setSearching(true);
    setError("");
    try {
      const data = await searchExistingPurchases({
        query: search.query,
        fkCentralUnit: search.fk_central_unit,
      });
      setExistingRows(data.items || []);
    } catch (err) {
      setError(err.message);
    } finally {
      setSearching(false);
    }
  }

  async function handleCheckPurchase(event) {
    event.preventDefault();
    if (!canCheck) return;

    setLoading(true);
    setError("");
    setResult(null);
    try {
      const data = await checkPurchase({
        product_name: form.product_name.trim(),
        fk_central_unit: Number(form.fk_central_unit),
        order_id: form.order_id.trim(),
        supply_order_date: form.supply_order_date,
      });
      setResult(data);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  }

  return (
    <main className="shell">
      <header className="topbar">
        <div>
          <h1>Duplicate Purchase Checker</h1>
          <p>Check a new product name or select an existing purchase from the database.</p>
        </div>
        <a className="docsLink" href="/docs">
          <FileSearch size={17} />
          API docs
        </a>
      </header>

      <section className="workspace">
        <section className="leftColumn">
          <form className="panel inputPanel" onSubmit={handleCheckPurchase}>
            <div className="panelHeader">
              <h2>Purchase Input</h2>
              <button type="button" className="ghostButton" onClick={resetInput}>
                <RefreshCcw size={16} />
                Clear
              </button>
            </div>

            {selectedRow && (
              <div className="selectedNote">Selected bill {selectedRow.bill_id} from existing data</div>
            )}

            <label>
              Product name
              <textarea
                value={form.product_name}
                onChange={(event) => updateForm("product_name", event.target.value)}
                placeholder="Example: rubber, eraser, cotton sports t-shirt"
                rows={3}
              />
            </label>

            <div className="formGrid">
              <label>
                Central unit
                <input
                  value={form.fk_central_unit}
                  onChange={(event) => updateForm("fk_central_unit", event.target.value)}
                  placeholder="4606"
                  inputMode="numeric"
                />
              </label>
              <label>
                Supply order date
                <input
                  type="date"
                  value={form.supply_order_date}
                  onChange={(event) => updateForm("supply_order_date", event.target.value)}
                />
              </label>
            </div>

            <label>
              Order ID
              <input
                value={form.order_id}
                onChange={(event) => updateForm("order_id", event.target.value)}
                placeholder="NEW_ORDER_001"
              />
            </label>

            <button className="primaryButton" disabled={!canCheck || loading}>
              <ShieldCheck size={18} />
              {loading ? "Checking..." : "Check duplicate"}
            </button>
          </form>

          {error && <div className="errorBox">{error}</div>}
          <ResultPanel result={result} />
        </section>

        <section className="panel existingPanel">
          <div className="panelHeader">
            <h2>Existing Data</h2>
            <span className="muted">{existingRows.length} rows</span>
          </div>

          <form className="searchBar" onSubmit={handleSearchExisting}>
            <input
              value={search.query}
              onChange={(event) => setSearch((current) => ({ ...current, query: event.target.value }))}
              placeholder="Search existing product name"
            />
            <input
              value={search.fk_central_unit}
              onChange={(event) =>
                setSearch((current) => ({ ...current, fk_central_unit: event.target.value }))
              }
              placeholder="Central unit"
              inputMode="numeric"
            />
            <button className="secondaryButton" disabled={searching}>
              <Search size={17} />
              {searching ? "Searching..." : "Search"}
            </button>
          </form>

          <ExistingRowsTable rows={existingRows} onSelect={selectExisting} />
        </section>
      </section>
    </main>
  );
}

function ExistingRowsTable({ rows, onSelect }) {
  return (
    <div className="tableWrap">
      <table>
        <thead>
          <tr>
            <th>Product</th>
            <th>Unit</th>
            <th>Date</th>
            <th>Order</th>
            <th></th>
          </tr>
        </thead>
        <tbody>
          {rows.length === 0 ? (
            <tr>
              <td colSpan="5" className="emptyCell">
                Search by product name or central unit to select existing data.
              </td>
            </tr>
          ) : (
            rows.map((row) => (
              <tr key={`${row.bill_id}-${row.product_name}`}>
                <td>{row.product_name}</td>
                <td>{row.fk_central_unit}</td>
                <td>{row.supply_order_date}</td>
                <td>{row.order_id || "-"}</td>
                <td>
                  <button type="button" className="smallButton" onClick={() => onSelect(row)}>
                    Use
                  </button>
                </td>
              </tr>
            ))
          )}
        </tbody>
      </table>
    </div>
  );
}

function ResultPanel({ result }) {
  if (!result) {
    return (
      <section className="panel resultPanel mutedPanel">
        Run a check to see duplicate status, similarity score, and conflicting bills.
      </section>
    );
  }

  return (
    <section className="panel resultPanel">
      <div className="resultHeader">
        <div>
          <h2>Result</h2>
          <p>{result.reason}</p>
        </div>
        <span className={`statusBadge ${result.decision}`}>
          {decisionIcon(result.decision)}
          {decisionLabel(result.decision)}
        </span>
      </div>

      <div className="metrics">
        <Metric label={result.rerank?.enabled ? "Best rerank" : "Best similarity"} value={result.best_similarity_score} />
        <Metric label="Candidate rows" value={result.candidate_count} />
        <Metric label="Duplicate threshold" value={result.thresholds.duplicate} />
        <Metric label="Review threshold" value={result.thresholds.manual_review} />
      </div>
      {result.rerank?.enabled && (
        <div className="rerankNote">
          Reranking top {result.rerank.top_k} candidates with {result.rerank.model}
        </div>
      )}

      <LlmMatchPanel match={result.llm_same_product} />

      <div className="resultSectionTitle">
        <h3>Conflicting Bills</h3>
        <span className="muted">{result.conflicting_bills.length} matches</span>
      </div>
      <div className="tableWrap resultTableWrap">
        <table>
          <thead>
            <tr>
              <th>Product</th>
              <th>{result.rerank?.enabled ? "Rerank" : "Similarity"}</th>
              {result.rerank?.enabled && <th>Embedding</th>}
              <th>Date</th>
              <th>Order</th>
              <th>Transaction</th>
            </tr>
          </thead>
          <tbody>
            {result.conflicting_bills.length === 0 ? (
              <tr>
                <td colSpan={result.rerank?.enabled ? "6" : "5"} className="emptyCell">
                  No similar historical purchases found.
                </td>
              </tr>
            ) : (
              result.conflicting_bills.map((bill) => (
                <tr key={`${bill.bill_id}-${bill.product_name}`}>
                  <td>{bill.product_name}</td>
                  <td>
                    {bill.similarity_score}
                    {bill.llm_same_product !== undefined && (
                      <div className={bill.llm_same_product ? "llmYes" : "llmNo"}>
                        LLM: {bill.llm_same_product ? "Same product" : "Different"}
                      </div>
                    )}
                  </td>
                  {result.rerank?.enabled && <td>{bill.embedding_similarity_score ?? "-"}</td>}
                  <td>{bill.supply_order_date}</td>
                  <td>{bill.order_id || "-"}</td>
                  <td>{bill.transaction_id || "-"}</td>
                </tr>
              ))
            )}
          </tbody>
        </table>
      </div>
    </section>
  );
}

function LlmMatchPanel({ match }) {
  if (!match) return null;

  return (
    <section className={`categoryMatch ${match.available ? "" : "categoryMatchMuted"}`}>
      <div>
        <span>LLM same-product check</span>
        <strong>{match.available ? "Completed" : "Unavailable"}</strong>
      </div>
      <div>
        <span>Model</span>
        <strong>{match.model || "-"}</strong>
      </div>
      <div>
        <span>Confirmed matches</span>
        <strong>{match.available ? match.matches.filter((item) => item.same_product).length : "-"}</strong>
      </div>
      <p>
        {match.available
          ? "The LLM compared the final reranked product names; only confirmed same products remain below."
          : match.reason}
      </p>
    </section>
  );
}

function Metric({ label, value }) {
  return (
    <div>
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  );
}

createRoot(document.getElementById("root")).render(<App />);
