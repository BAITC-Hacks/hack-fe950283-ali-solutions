const assert = require("node:assert/strict");
const test = require("node:test");
const {restore, toCsv} = require("../viewer/review.js");

const identifier = "100000000000000001";
const knownIds = new Set([identifier]);

test("review persistence keeps full int64 identifiers as strings", () => {
  const records = restore(JSON.stringify([{gid:identifier, status:"requested", note:"Выписка"}]), knownIds);
  assert.equal(records.get(identifier).gid, identifier);
  assert.equal(records.get(identifier).status, "requested");
  assert.equal(records.get(identifier).note, "Выписка");
});

test("review import rejects unknown and rounded numeric identifiers", () => {
  const records = restore(JSON.stringify([{gid: Number(identifier)}, {gid:"unknown"}]), knownIds);
  assert.equal(records.size, 0);
});

test("review import normalizes status, caps notes, and handles duplicates", () => {
  const records = restore(JSON.stringify([{gid:identifier}, {gid:identifier, status:"__proto__", note:"x".repeat(3000)}]), knownIds);
  assert.equal(records.size, 1);
  assert.equal(records.get(identifier).status, "new");
  assert.equal(records.get(identifier).note.length, 2000);
  assert.throws(() => restore("{}", knownIds));
});

test("CSV quotes notes, preserves gid, and neutralizes spreadsheet formulas", () => {
  const records = restore(JSON.stringify([{gid:identifier, note:'=HYPERLINK("bad")\nnext'}]), knownIds);
  const nodes = new Map([[identifier, {id:identifier, r:"transit", p:.75, ev:'Text, "quoted"', req:["Выписка"]}]]);
  const output = toCsv(records, nodes);
  assert.ok(output.startsWith("\uFEFF"));
  assert.ok(output.includes('"' + identifier + '"'));
  assert.ok(output.includes('"\'=HYPERLINK(""bad"")\nnext"'));
  assert.ok(output.includes('"Text, ""quoted"""'));
});
