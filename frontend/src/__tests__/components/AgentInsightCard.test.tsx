import { render, screen } from "@testing-library/react";
import { describe, it, expect } from "vitest";
import AgentInsightCard from "@/components/data-display/AgentInsightCard";

describe("AgentInsightCard", () => {
  it("renders title and content", () => {
    render(
      <AgentInsightCard
        title="Risk Assessment"
        content="High risk detected for 2,420 accounts."
      />,
    );
    expect(screen.getByText("Risk Assessment")).toBeInTheDocument();
    expect(
      screen.getByText("High risk detected for 2,420 accounts."),
    ).toBeInTheDocument();
  });

  it("renders AI Insight label", () => {
    render(
      <AgentInsightCard title="Summary" content="Test content" />,
    );
    expect(screen.getByText("AI Insight")).toBeInTheDocument();
  });

  it("renders timestamp when provided", () => {
    render(
      <AgentInsightCard
        title="Summary"
        content="Test"
        timestamp="5 min ago"
      />,
    );
    expect(screen.getByText("5 min ago")).toBeInTheDocument();
  });

  it("omits timestamp when not provided", () => {
    const { container } = render(
      <AgentInsightCard title="Summary" content="Test" />,
    );
    // Should not contain any element with timestamp-like content
    expect(container.textContent).not.toContain("min ago");
  });
});
