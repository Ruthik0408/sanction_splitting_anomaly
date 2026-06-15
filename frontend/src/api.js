const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || "";

async function request(path, options = {}) {
  const response = await fetch(`${API_BASE_URL}${path}`, options);
  const data = await response.json().catch(() => ({}));
  if (!response.ok) {
    throw new Error(data.detail || `Request failed with status ${response.status}`);
  }
  return data;
}

export function searchExistingPurchases({ query, fkCentralUnit, limit = 30 }) {
  const params = new URLSearchParams();
  if (query.trim()) params.set("query", query.trim());
  if (String(fkCentralUnit).trim()) params.set("fk_central_unit", String(fkCentralUnit).trim());
  params.set("limit", String(limit));
  return request(`/existing_purchases?${params.toString()}`);
}

export function checkPurchase(payload) {
  return request("/check_purchase", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
}
