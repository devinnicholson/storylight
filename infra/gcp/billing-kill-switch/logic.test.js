"use strict";

const assert = require("node:assert/strict");
const test = require("node:test");
const {BudgetMessageError, evaluateBudgetNotification} = require("./logic");

const expected = {
  billingAccountId: "000000-000000-000000",
  budgetAmount: "10",
  budgetDisplayName: "Bookforge emergency billing disconnect",
  budgetId: "budget-id",
  projectId: "your-gcp-project",
};

function event(costAmount, overrides = {}) {
  const payload = {
    budgetAmount: 10,
    budgetDisplayName: expected.budgetDisplayName,
    costAmount,
    currencyCode: "USD",
    ...overrides.payload,
  };
  return {
    data: {
      message: {
        attributes: {
          billingAccountId: expected.billingAccountId,
          budgetId: expected.budgetId,
          ...overrides.attributes,
        },
        data: Buffer.from(JSON.stringify(payload)).toString("base64"),
      },
    },
  };
}

test("ignores a valid notification below the gross-cost threshold", () => {
  assert.deepEqual(evaluateBudgetNotification(event(9.99), expected), {
    action: "ignore",
    budgetAmount: 10,
    costAmount: 9.99,
    projectName: "projects/your-gcp-project",
  });
});

test("disconnects billing at the exact gross-cost threshold", () => {
  assert.equal(evaluateBudgetNotification(event(10), expected).action, "disable-billing");
});

test("disconnects billing above the gross-cost threshold", () => {
  assert.equal(evaluateBudgetNotification(event(12.34), expected).action, "disable-billing");
});

test("rejects a notification from another budget", () => {
  assert.throws(
    () =>
      evaluateBudgetNotification(
        event(10, {attributes: {budgetId: "wrong-budget"}}),
        expected,
      ),
    BudgetMessageError,
  );
});

test("rejects a notification whose amount was changed", () => {
  assert.throws(
    () => evaluateBudgetNotification(event(10, {payload: {budgetAmount: 15}}), expected),
    BudgetMessageError,
  );
});

test("rejects malformed Pub/Sub data", () => {
  assert.throws(
    () => evaluateBudgetNotification({data: {message: {data: "not-json"}}}, expected),
    BudgetMessageError,
  );
});
