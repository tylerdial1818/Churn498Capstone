import { ResponsiveContainer, PieChart, Pie, Cell, Tooltip } from "recharts";

interface DonutChartProps {
  data: { name: string; value: number; color: string }[];
  height?: number;
  innerLabel?: string;
  innerValue?: string;
}

export default function DonutChart({
  data,
  height = 240,
  innerLabel,
  innerValue,
}: DonutChartProps) {
  return (
    <div className="relative">
      <ResponsiveContainer width="100%" height={height}>
        <PieChart>
          <Pie
            data={data}
            cx="50%"
            cy="50%"
            innerRadius="60%"
            outerRadius="80%"
            paddingAngle={2}
            dataKey="value"
          >
            {data.map((entry, index) => (
              <Cell key={index} fill={entry.color} stroke="none" />
            ))}
          </Pie>
          <Tooltip
            contentStyle={{
              background: "#12131a",
              border: "1px solid #2a2b36",
              borderRadius: 8,
              fontSize: 12,
            }}
            formatter={(value: number) => [value.toLocaleString(), ""]}
          />
        </PieChart>
      </ResponsiveContainer>
      {innerLabel && (
        <div className="pointer-events-none absolute inset-0 flex flex-col items-center justify-center">
          <span className="text-2xl font-semibold text-text-primary">{innerValue}</span>
          <span className="text-xs text-text-muted">{innerLabel}</span>
        </div>
      )}
    </div>
  );
}
