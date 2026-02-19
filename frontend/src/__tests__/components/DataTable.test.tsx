import { render, screen, fireEvent } from "@testing-library/react";
import { describe, it, expect, vi } from "vitest";
import DataTable from "@/components/data-display/DataTable";
import type { Column } from "@/components/data-display/DataTable";

interface TestRow {
  id: string;
  name: string;
  value: number;
}

const columns: Column<TestRow>[] = [
  { key: "name", header: "Name", render: (r) => r.name },
  { key: "value", header: "Value", render: (r) => r.value.toString() },
];

const testData: TestRow[] = [
  { id: "1", name: "Alice", value: 100 },
  { id: "2", name: "Bob", value: 200 },
  { id: "3", name: "Charlie", value: 300 },
];

describe("DataTable", () => {
  it("renders columns and rows", () => {
    render(
      <DataTable
        columns={columns}
        data={testData}
        keyExtractor={(r) => r.id}
      />,
    );
    expect(screen.getByText("Name")).toBeInTheDocument();
    expect(screen.getByText("Value")).toBeInTheDocument();
    expect(screen.getByText("Alice")).toBeInTheDocument();
    expect(screen.getByText("Bob")).toBeInTheDocument();
    expect(screen.getByText("Charlie")).toBeInTheDocument();
  });

  it("renders correct number of rows", () => {
    const { container } = render(
      <DataTable
        columns={columns}
        data={testData}
        keyExtractor={(r) => r.id}
      />,
    );
    const rows = container.querySelectorAll("tbody tr");
    expect(rows.length).toBe(3);
  });

  it("shows empty state when no data", () => {
    render(
      <DataTable
        columns={columns}
        data={[]}
        keyExtractor={(r: TestRow) => r.id}
        emptyMessage="Nothing here"
      />,
    );
    expect(screen.getByText("Nothing here")).toBeInTheDocument();
  });

  it("calls onRowClick when row is clicked", () => {
    const onClick = vi.fn();
    render(
      <DataTable
        columns={columns}
        data={testData}
        keyExtractor={(r) => r.id}
        onRowClick={onClick}
      />,
    );
    fireEvent.click(screen.getByText("Alice"));
    expect(onClick).toHaveBeenCalledWith(testData[0]);
  });

  it("shows default empty message", () => {
    render(
      <DataTable
        columns={columns}
        data={[]}
        keyExtractor={(r: TestRow) => r.id}
      />,
    );
    expect(screen.getByText("No data available")).toBeInTheDocument();
  });
});
