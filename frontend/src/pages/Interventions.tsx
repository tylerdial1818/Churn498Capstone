import { useState } from "react";
import PageShell from "@/components/layout/PageShell";
import FilterBar from "@/components/data-display/FilterBar";
import DataTable from "@/components/data-display/DataTable";
import StatusBadge from "@/components/data-display/StatusBadge";
import LoadingSkeleton from "@/components/data-display/LoadingSkeleton";
import EmptyState from "@/components/data-display/EmptyState";
import { useInterventions, useUpdateInterventionStatus } from "@/hooks/useInterventions";
import type { InterventionDraft } from "@/types/api";
import type { Column } from "@/components/data-display/DataTable";

const STATUS_FILTERS = [
  { label: "All", value: "" },
  { label: "Pending", value: "pending" },
  { label: "Approved", value: "approved" },
  { label: "Sent", value: "sent" },
  { label: "Rejected", value: "rejected" },
];

export default function Interventions() {
  const [statusFilter, setStatusFilter] = useState("");
  const [selected, setSelected] = useState<InterventionDraft | null>(null);
  const { data: interventions, isLoading } = useInterventions(statusFilter || undefined);
  const updateStatus = useUpdateInterventionStatus();

  const COLUMNS: Column<InterventionDraft>[] = [
    {
      key: "account_id",
      header: "Account",
      render: (r) => <span className="font-medium text-text-primary">{r.account_id}</span>,
    },
    {
      key: "strategy",
      header: "Strategy",
      render: (r) => (
        <span className="text-text-secondary capitalize">{r.strategy.replace(/_/g, " ")}</span>
      ),
    },
    {
      key: "subject",
      header: "Subject",
      render: (r) => (
        <span className="max-w-xs truncate text-text-secondary">{r.subject}</span>
      ),
    },
    {
      key: "status",
      header: "Status",
      render: (r) => <StatusBadge status={r.status} />,
    },
    {
      key: "updated_at",
      header: "Updated",
      render: (r) => (
        <span className="text-xs text-text-muted">
          {new Date(r.updated_at).toLocaleDateString()}
        </span>
      ),
    },
  ];

  return (
    <PageShell
      title="Interventions"
      subtitle="AI-drafted retention emails"
      actions={
        <FilterBar options={STATUS_FILTERS} value={statusFilter} onChange={setStatusFilter} />
      }
    >
      <div className="grid gap-6 lg:grid-cols-3">
        {/* Table */}
        <div className="lg:col-span-2">
          {isLoading ? (
            <LoadingSkeleton variant="row" count={6} />
          ) : interventions && interventions.length > 0 ? (
            <DataTable
              columns={COLUMNS}
              data={interventions}
              keyExtractor={(r) => r.id}
              onRowClick={(r) => setSelected(r)}
            />
          ) : (
            <EmptyState
              title="No interventions yet"
              description="Draft interventions from the Prescriptions page."
            />
          )}
        </div>

        {/* Detail panel */}
        <div>
          {selected ? (
            <div className="space-y-4">
              {/* Header */}
              <div className="rounded-card border border-border bg-surface p-card">
                <div className="flex items-start justify-between">
                  <div>
                    <h3 className="font-medium text-text-primary">{selected.account_id}</h3>
                    <p className="text-xs text-text-muted capitalize">
                      {selected.strategy.replace(/_/g, " ")}
                    </p>
                  </div>
                  <StatusBadge status={selected.status} />
                </div>

                {/* Actions */}
                {selected.status === "pending" && (
                  <div className="mt-4 flex gap-2">
                    <button
                      onClick={() =>
                        updateStatus.mutate(
                          { id: selected.id, status: "approved" },
                          {
                            onSuccess: (updated) => setSelected(updated),
                          },
                        )
                      }
                      disabled={updateStatus.isPending}
                      className="flex-1 rounded-button bg-accent px-3 py-2 text-sm font-medium text-white transition-colors hover:bg-accent-hover disabled:opacity-50"
                    >
                      Approve
                    </button>
                    <button
                      onClick={() =>
                        updateStatus.mutate(
                          { id: selected.id, status: "rejected" },
                          {
                            onSuccess: (updated) => setSelected(updated),
                          },
                        )
                      }
                      disabled={updateStatus.isPending}
                      className="flex-1 rounded-button border border-border px-3 py-2 text-sm font-medium text-text-secondary transition-colors hover:bg-surface-hover disabled:opacity-50"
                    >
                      Reject
                    </button>
                  </div>
                )}
              </div>

              {/* Email Preview */}
              <div className="rounded-card border border-border bg-surface p-card">
                <h4 className="mb-2 text-xs font-medium uppercase tracking-wider text-text-muted">
                  Email Preview
                </h4>
                <p className="mb-3 text-sm font-medium text-text-primary">{selected.subject}</p>
                <div
                  className="prose prose-sm prose-invert max-w-none text-xs leading-relaxed text-text-secondary"
                  dangerouslySetInnerHTML={{ __html: selected.body_html }}
                />
              </div>

              {/* Agent Rationale */}
              <div className="rounded-card border border-accent/20 bg-accent-muted p-card">
                <div className="mb-2 flex items-center gap-2">
                  <svg
                    className="h-4 w-4 text-accent"
                    fill="none"
                    viewBox="0 0 24 24"
                    stroke="currentColor"
                    strokeWidth={2}
                  >
                    <path
                      strokeLinecap="round"
                      strokeLinejoin="round"
                      d="M9.813 15.904L9 18.75l-.813-2.846a4.5 4.5 0 00-3.09-3.09L2.25 12l2.846-.813a4.5 4.5 0 003.09-3.09L9 5.25l.813 2.846a4.5 4.5 0 003.09 3.09L15.75 12l-2.846.813a4.5 4.5 0 00-3.09 3.09z"
                    />
                  </svg>
                  <span className="text-xs font-semibold uppercase tracking-wider text-accent">
                    AI Rationale
                  </span>
                </div>
                <p className="text-xs leading-relaxed text-text-secondary">
                  {selected.agent_rationale}
                </p>
              </div>

              {/* Timestamps */}
              <div className="flex justify-between text-xs text-text-muted">
                <span>Created: {new Date(selected.created_at).toLocaleString()}</span>
                <span>Updated: {new Date(selected.updated_at).toLocaleString()}</span>
              </div>
            </div>
          ) : (
            <div className="flex h-64 items-center justify-center rounded-card border border-dashed border-border bg-surface">
              <p className="text-sm text-text-muted">Select an intervention to preview</p>
            </div>
          )}
        </div>
      </div>
    </PageShell>
  );
}
