(function (root) {
  "use strict";
  const statuses = {new: "К проверке", requested: "Запрошены данные", reviewed: "Рассмотрен"};

  function restore(serialized, knownIds) {
    const saved = JSON.parse(serialized || "[]");
    if (!Array.isArray(saved)) throw new Error("Неверный формат списка проверки");
    const records = new Map();
    for (const record of saved) {
      if (!record || typeof record.gid !== "string" || !knownIds.has(record.gid)) continue;
      records.set(record.gid, {
        gid: record.gid,
        status: Object.prototype.hasOwnProperty.call(statuses, record.status) ? record.status : "new",
        note: typeof record.note === "string" ? record.note.slice(0, 2000) : "",
      });
    }
    return records;
  }

  function csvCell(value) {
    let text = String(value ?? "");
    if (/^[\s]*[=+@-]/.test(text) || /^[\t\r\n]/.test(text)) text = "'" + text;
    return '"' + text.replaceAll('"', '""') + '"';
  }

  function toCsv(records, byId) {
    const rows = [["gid", "role", "priority_score", "status", "note", "evidence", "requested_data"]];
    for (const record of records.values()) {
      const node = byId.get(record.gid);
      if (!node) continue;
      rows.push([node.id, node.r, node.p, statuses[record.status], record.note, node.ev, node.req.join("; ")]);
    }
    return "\uFEFF" + rows.map(row => row.map(csvCell).join(",")).join("\r\n") + "\r\n";
  }

  const api = {statuses, restore, toCsv};
  if (typeof module !== "undefined" && module.exports) module.exports = api;
  else root.MoneyGraphReview = api;
})(globalThis);
