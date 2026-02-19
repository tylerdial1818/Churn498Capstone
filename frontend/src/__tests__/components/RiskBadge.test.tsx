import { render, screen } from "@testing-library/react";
import { describe, it, expect } from "vitest";
import RiskBadge from "@/components/data-display/RiskBadge";

describe("RiskBadge", () => {
  it("renders high tier text", () => {
    render(<RiskBadge tier="high" />);
    expect(screen.getByText("high")).toBeInTheDocument();
  });

  it("renders medium tier text", () => {
    render(<RiskBadge tier="medium" />);
    expect(screen.getByText("medium")).toBeInTheDocument();
  });

  it("renders low tier text", () => {
    render(<RiskBadge tier="low" />);
    expect(screen.getByText("low")).toBeInTheDocument();
  });

  it("applies correct color class for high tier", () => {
    const { container } = render(<RiskBadge tier="high" />);
    const badge = container.firstChild as HTMLElement;
    expect(badge.className).toContain("risk-high");
  });

  it("applies correct color class for medium tier", () => {
    const { container } = render(<RiskBadge tier="medium" />);
    const badge = container.firstChild as HTMLElement;
    expect(badge.className).toContain("risk-medium");
  });
});
