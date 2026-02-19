interface AgentInsightCardProps {
  title: string;
  content: string;
  timestamp?: string;
}

export default function AgentInsightCard({ title, content, timestamp }: AgentInsightCardProps) {
  return (
    <div className="rounded-card border border-accent/20 bg-accent-muted p-card">
      <div className="mb-2 flex items-center gap-2">
        <div className="flex h-5 w-5 items-center justify-center rounded bg-accent/20">
          <svg className="h-3 w-3 text-accent" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
            <path strokeLinecap="round" strokeLinejoin="round" d="M9.813 15.904L9 18.75l-.813-2.846a4.5 4.5 0 00-3.09-3.09L2.25 12l2.846-.813a4.5 4.5 0 003.09-3.09L9 5.25l.813 2.846a4.5 4.5 0 003.09 3.09L15.75 12l-2.846.813a4.5 4.5 0 00-3.09 3.09z" />
          </svg>
        </div>
        <span className="text-xs font-semibold uppercase tracking-wider text-accent">AI Insight</span>
        {timestamp && <span className="ml-auto text-xs text-text-muted">{timestamp}</span>}
      </div>
      <h4 className="text-sm font-medium text-text-primary">{title}</h4>
      <p className="mt-1 text-xs leading-relaxed text-text-secondary">{content}</p>
    </div>
  );
}
