import { render, screen } from "@testing-library/react";
import { describe, it, expect } from "vitest";
import KPICard from "@/components/data-display/KPICard";

describe("KPICard", () => {
  it("renders label and value", () => {
    render(<KPICard label="Total Accounts" value="60,000" />);
    expect(screen.getByText("Total Accounts")).toBeInTheDocument();
    expect(screen.getByText("60,000")).toBeInTheDocument();
  });

  it("renders subtitle when provided", () => {
    render(<KPICard label="Active" value="49,684" subtitle="subscribers" />);
    expect(screen.getByText("subscribers")).toBeInTheDocument();
  });

  it("hides subtitle when not provided", () => {
    const { container } = render(<KPICard label="Test" value="100" />);
    const subtitles = container.querySelectorAll("p");
    // Should have label + value only (2 p tags), no subtitle
    expect(subtitles.length).toBe(2);
  });

  it("renders trend when provided", () => {
    render(
      <KPICard
        label="Churn Rate"
        value="4.9%"
        trend={{ value: 0.5, label: "vs last month" }}
      />,
    );
    expect(screen.getByText("+0.5%")).toBeInTheDocument();
    expect(screen.getByText("vs last month")).toBeInTheDocument();
  });

  it("shows negative trend without plus sign", () => {
    render(
      <KPICard
        label="Churn Rate"
        value="4.9%"
        trend={{ value: -0.3, label: "vs last month" }}
      />,
    );
    expect(screen.getByText("-0.3%")).toBeInTheDocument();
  });

  it("applies accent variant styling", () => {
    const { container } = render(
      <KPICard label="MRR" value="$33,880" variant="accent" />,
    );
    const card = container.firstChild as HTMLElement;
    expect(card.className).toContain("accent");
  });
});
