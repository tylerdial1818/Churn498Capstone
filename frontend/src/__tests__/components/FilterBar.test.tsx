import { render, screen, fireEvent } from "@testing-library/react";
import { describe, it, expect, vi } from "vitest";
import FilterBar from "@/components/data-display/FilterBar";

const options = [
  { label: "All", value: "" },
  { label: "High", value: "high" },
  { label: "Medium", value: "medium" },
];

describe("FilterBar", () => {
  it("renders all filter options", () => {
    render(<FilterBar options={options} value="" onChange={() => {}} />);
    expect(screen.getByText("All")).toBeInTheDocument();
    expect(screen.getByText("High")).toBeInTheDocument();
    expect(screen.getByText("Medium")).toBeInTheDocument();
  });

  it("calls onChange when option is clicked", () => {
    const onChange = vi.fn();
    render(<FilterBar options={options} value="" onChange={onChange} />);
    fireEvent.click(screen.getByText("High"));
    expect(onChange).toHaveBeenCalledWith("high");
  });

  it("highlights active option", () => {
    const { container } = render(
      <FilterBar options={options} value="high" onChange={() => {}} />,
    );
    const buttons = container.querySelectorAll("button");
    // "High" button (index 1) should have accent styling
    expect(buttons[1].className).toContain("bg-accent");
  });
});
