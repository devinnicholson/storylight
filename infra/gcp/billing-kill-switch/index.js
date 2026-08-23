"use strict";

const functions = require("@google-cloud/functions-framework");
const {CloudBillingClient} = require("@google-cloud/billing");
const {BudgetMessageError, evaluateBudgetNotification} = require("./logic");

const billing = new CloudBillingClient();

function expectedConfiguration(environment = process.env) {
  return {
    billingAccountId: environment.EXPECTED_BILLING_ACCOUNT_ID,
    budgetAmount: environment.EXPECTED_BUDGET_AMOUNT,
    budgetDisplayName: environment.EXPECTED_BUDGET_DISPLAY_NAME,
    budgetId: environment.EXPECTED_BUDGET_ID,
    projectId: environment.EXPECTED_PROJECT_ID,
  };
}

function createStopBillingHandler({billingClient, environment, logger} = {}) {
  const client = billingClient ?? billing;
  const env = environment ?? process.env;
  const log = logger ?? console;

  return async function stopBilling(cloudEvent) {
    let decision;
    try {
      decision = evaluateBudgetNotification(cloudEvent, expectedConfiguration(env));
    } catch (error) {
      if (error instanceof BudgetMessageError) {
        log.error("Rejected billing notification:", error.message);
        return;
      }
      throw error;
    }

    if (decision.action === "ignore") {
      log.log(
        `Gross project cost is $${decision.costAmount.toFixed(2)}; ` +
          `$${decision.budgetAmount.toFixed(2)} emergency threshold not reached.`,
      );
      return;
    }

    log.error(
      `Emergency threshold reached at $${decision.costAmount.toFixed(2)}. ` +
        `Disabling billing for ${decision.projectName}.`,
    );
    await client.updateProjectBillingInfo({
      name: decision.projectName,
      projectBillingInfo: {billingAccountName: ""},
    });
    log.error(`Billing disabled for ${decision.projectName}.`);
  };
}

const stopBilling = createStopBillingHandler();
functions.cloudEvent("stopBilling", stopBilling);

module.exports = {createStopBillingHandler, expectedConfiguration, stopBilling};
