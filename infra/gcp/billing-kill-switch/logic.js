"use strict";

class BudgetMessageError extends Error {
  constructor(message) {
    super(message);
    this.name = "BudgetMessageError";
  }
}

function requiredString(value, label) {
  if (typeof value !== "string" || value.length === 0) {
    throw new BudgetMessageError(`${label} is required`);
  }
  return value;
}

function requiredMoney(value, label) {
  const number = Number(value);
  if (!Number.isFinite(number) || number < 0) {
    throw new BudgetMessageError(`${label} must be a non-negative number`);
  }
  return number;
}

function parseBudgetNotification(cloudEvent) {
  const message = cloudEvent?.data?.message;
  if (!message || typeof message.data !== "string") {
    throw new BudgetMessageError("Pub/Sub message data is required");
  }

  let payload;
  try {
    const json = Buffer.from(message.data, "base64").toString("utf8");
    payload = JSON.parse(json);
  } catch (error) {
    throw new BudgetMessageError(`Budget payload is not valid JSON: ${error.message}`);
  }

  if (!payload || typeof payload !== "object" || Array.isArray(payload)) {
    throw new BudgetMessageError("Budget payload must be an object");
  }

  return {
    attributes: message.attributes ?? {},
    payload,
  };
}

function evaluateBudgetNotification(cloudEvent, expected) {
  const expectedProjectId = requiredString(expected.projectId, "expected project ID");
  const expectedBillingAccountId = requiredString(
    expected.billingAccountId,
    "expected billing account ID",
  );
  const expectedBudgetId = requiredString(expected.budgetId, "expected budget ID");
  const expectedBudgetDisplayName = requiredString(
    expected.budgetDisplayName,
    "expected budget display name",
  );
  const expectedBudgetAmount = requiredMoney(expected.budgetAmount, "expected budget amount");
  const {attributes, payload} = parseBudgetNotification(cloudEvent);

  if (attributes.billingAccountId !== expectedBillingAccountId) {
    throw new BudgetMessageError("Notification came from an unexpected billing account");
  }
  if (attributes.budgetId !== expectedBudgetId) {
    throw new BudgetMessageError("Notification came from an unexpected budget");
  }
  if (payload.budgetDisplayName !== expectedBudgetDisplayName) {
    throw new BudgetMessageError("Notification has an unexpected budget display name");
  }
  if (payload.currencyCode !== "USD") {
    throw new BudgetMessageError("Notification currency must be USD");
  }

  const budgetAmount = requiredMoney(payload.budgetAmount, "notification budget amount");
  if (Math.abs(budgetAmount - expectedBudgetAmount) > 0.000001) {
    throw new BudgetMessageError("Notification has an unexpected budget amount");
  }

  const costAmount = requiredMoney(payload.costAmount, "notification cost amount");
  return {
    action: costAmount >= expectedBudgetAmount ? "disable-billing" : "ignore",
    budgetAmount,
    costAmount,
    projectName: `projects/${expectedProjectId}`,
  };
}

module.exports = {
  BudgetMessageError,
  evaluateBudgetNotification,
  parseBudgetNotification,
};
