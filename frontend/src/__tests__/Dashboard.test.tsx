import { render, screen } from "@testing-library/react";
import { describe, it, expect, vi } from "vitest";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { MemoryRouter } from "react-router-dom";
import Dashboard from "@/pages/Dashboard";

// Mock the hooks
vi.mock("@/hooks/useDashboard", () => ({
  useKPIs: () => ({
    data: {
      total_accounts: 60000,
      active_subscribers: 49684,
      churned_accounts: 10316,
      churn_rate_30d: 0.049,
      high_risk_count: 2420,
      at_risk_mrr: 33880,
      cac: 45,
      retention_cost_per_save: 12.5,
    },
    isLoading: false,
  }),
  useTrends: () => ({
    data: [
      { label: "Jan", value: 385, predicted: 400 },
      { label: "Feb", value: 410, predicted: 420 },
    ],
    isLoading: false,
  }),
  useRiskDistribution: () => ({
    data: { low: 53760, medium: 3820, high: 2420 },
    isLoading: false,
  }),
  useActiveInactive: () => ({
    data: { active: 49684, inactive: 10316, recent_churn_30d: 685 },
    isLoading: false,
  }),
  useExecutiveSummary: () => ({
    data: {
      title: "Subscriber Health Summary",
      content: "The platform currently serves 49,684 active subscribers.",
      timestamp: "5 min ago",
    },
    isLoading: false,
  }),
}));

function renderWithProviders(ui: React.ReactElement) {
  const queryClient = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  });
  return render(
    <QueryClientProvider client={queryClient}>
      <MemoryRouter>{ui}</MemoryRouter>
    </QueryClientProvider>,
  );
}

describe("Dashboard", () => {
  it("renders the page title", () => {
    renderWithProviders(<Dashboard />);
    expect(screen.getByText("Dashboard")).toBeInTheDocument();
  });

  it("renders KPI cards with data", () => {
    renderWithProviders(<Dashboard />);
    // Values may appear in both KPI cards and chart labels, so use getAllByText
    expect(screen.getAllByText("Total Accounts").length).toBeGreaterThanOrEqual(1);
    expect(screen.getAllByText("60,000").length).toBeGreaterThanOrEqual(1);
    expect(screen.getByText("Churn Rate (30d)")).toBeInTheDocument();
    expect(screen.getAllByText("High Risk").length).toBeGreaterThanOrEqual(1);
  });

  it("renders executive summary", () => {
    renderWithProviders(<Dashboard />);
    expect(
      screen.getByText("The platform currently serves 49,684 active subscribers."),
    ).toBeInTheDocument();
  });

  it("renders at-risk MRR", () => {
    renderWithProviders(<Dashboard />);
    expect(screen.getByText("At-Risk MRR")).toBeInTheDocument();
    expect(screen.getByText("$33,880")).toBeInTheDocument();
  });
});
