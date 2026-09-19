// Unit tests for the shared timeline-batching helper.
//
//   node --test metainfer/tasks/sys_shell/static/components/utils.test.mjs
//
// CI runs pytest only (there is no JS runner configured), so these are not
// part of the CI gate — but utils.js is dependency-free, which is exactly why
// the helper lives there rather than inline in a view: it can be exercised
// here without a DOM or a bundler.

import test from "node:test";
import assert from "node:assert/strict";

import { takeTimelineBatch } from "./utils.js";

const ev = (ts, type = "x") => ({ ts, type });

test("first batch (cursor 0) is taken whole and sets the cursor", () => {
  const { fresh, since } = takeTimelineBatch([ev(100), ev(101), ev(102)], 0);
  assert.deepEqual(fresh.map((e) => e.ts), [100, 101, 102]);
  assert.equal(since, 102);
});

test("cursor advances to the newest timestamp in the batch", () => {
  const { since } = takeTimelineBatch([ev(100), ev(105), ev(103)], 90);
  assert.equal(since, 105);
});

test("the boundary event that ts >= since hands back is dropped, not re-appended", () => {
  // The endpoint filters with `ts >= since`, so asking from 102 returns 102 again.
  const { fresh, since } = takeTimelineBatch([ev(102), ev(103)], 102);
  assert.deepEqual(fresh.map((e) => e.ts), [103]);
  assert.equal(since, 103);
});

test("a batch with nothing new yields no fresh events and keeps the cursor", () => {
  const { fresh, since } = takeTimelineBatch([ev(100), ev(100)], 100);
  assert.deepEqual(fresh, []);
  assert.equal(since, 100);
});

test("repeated polling reconstructs the stream with no gaps and no duplicates", () => {
  // Model a run that writes events between polls: at poll N, only the first
  // `writtenByPoll[N]` events exist on disk.
  const stream = [ev(1), ev(2), ev(3), ev(4), ev(5)];
  const writtenByPoll = [2, 4, 5, 5];

  let cursor = 0;
  let seen = [];
  for (const written of writtenByPoll) {
    const onDisk = stream.filter((e) => e.ts <= written);
    const fromServer = onDisk.filter((e) => e.ts >= cursor); // endpoint semantics
    const { fresh, since } = takeTimelineBatch(fromServer, cursor);
    cursor = since;
    seen = seen.concat(fresh);
  }

  assert.deepEqual(seen.map((e) => e.ts), [1, 2, 3, 4, 5]);
});

test("malformed input never throws and never rewinds the cursor", () => {
  assert.deepEqual(takeTimelineBatch(null, 7), { fresh: [], since: 7 });
  assert.deepEqual(takeTimelineBatch(undefined, 7), { fresh: [], since: 7 });
  assert.deepEqual(takeTimelineBatch("nope", 7), { fresh: [], since: 7 });

  const { fresh, since } = takeTimelineBatch([null, { type: "no-ts" }, ev(9)], 7);
  assert.deepEqual(fresh.map((e) => e && e.ts), [9]);
  assert.equal(since, 9);
});
