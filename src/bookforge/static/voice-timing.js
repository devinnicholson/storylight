(function () {
  function summarize(events) {
    const last = (type, jobId) => events.findLast((event) => event.type === type
      && (jobId === undefined || event.jobId === jobId));
    const preview = last("preview_activated");
    const jobId = preview?.jobId;
    const response = jobId && last("generation_response", jobId);
    const requested = response?.submissionId && events.findLast((event) => event.type === "generation_requested"
      && event.submissionId === response.submissionId && event.ms <= response.ms);
    const completed = jobId && last("generation_completed", jobId);
    const interval = (start, end) => start && end && end.ms >= start.ms
      ? Math.round((end.ms - start.ms) * 10) / 10 : null;
    return {
      previewJobId: jobId ?? null,
      recordingToPreviewMs: interval(last("recording_started"), preview),
      detectedAudioToPreviewMs: interval(last("audio_activity"), preview),
      finishButtonToPreviewMs: interval(last("recording_stopped"), preview),
      requestToPreviewMs: interval(requested, preview),
      completionToPreviewMs: interval(completed, preview),
      connectionPreparationMs: interval(last("provider_preconnect_started"),
        last("provider_preconnect_completed")),
      connectionPreparationFailed: last("provider_preconnect_completed")?.failed ?? null,
    };
  }

  if (typeof module !== "undefined") module.exports = {summarize};
  if (typeof window !== "undefined") window.renderBookforgeVoiceTiming = (events) => {
    const output = document.getElementById("voiceTimingSummary");
    if (!output) return;
    output.closest("details").hidden = false;
    output.textContent = JSON.stringify({summary: summarize(events), events}, null, 2);
  };
})();
