"use strict";

const assert = require("node:assert/strict");
const test = require("node:test");
const {createStopBillingHandler} = require("./index");

const environment = {
  EXPECTED_BILLING_ACCOUNT_ID: "000000-000000-000000",
  EXPECTED_BUDGET_AMOUNT: "10",
  EXPECTED_BUDGET_DISPLAY_NAME: "Bookforge emergency billing disconnect",
  EXPECTED_BUDGET_ID: "budget-id",
  EXPECTED_PROJECT_ID: "your-gcp-project",
};

function event(costAmount) {
  const payload = {
    budgetAmount: 10,
    budgetDisplayName: environment.EXPECTED_BUDGET_DISPLAY_NAME,
    costAmount,
    currencyCode: "USD",
  };
  return {
    data: {
      message: {
        attributes: {
          billingAccountId: environment.EXPECTED_BILLING_ACCOUNT_ID,
          budgetId: environment.EXPECTED_BUDGET_ID,
        },
        data: Buffer.from(JSON.stringify(payload)).toString("base64"),
      },
    },
  };
}

function harness() {
  const requests = [];
  const handler = createStopBillingHandler({
    billingClient: {
      updateProjectBillingInfo: async request => requests.push(request),
    },
    environment,
    logger: {error() {}, log() {}},
  });
  return {handler, requests};
}

test("uses the current Billing SDK request field when the threshold is reached", async () => {
  const {handler, requests} = harness();
  await handler(event(10));
  assert.deepEqual(requests, [
    {
      name: "projects/your-gcp-project",
      projectBillingInfo: {billingAccountName: ""},
    },
  ]);
});

test("never calls Cloud Billing below the threshold", async () => {
  const {handler, requests} = harness();
  await handler(event(9.99));
  assert.deepEqual(requests, []);
});
