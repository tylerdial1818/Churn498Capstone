import {
  ResponsiveContainer,
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
} from "recharts";

interface TrendLineChartProps {
  data: { label: string; value: number; predicted?: number }[];
  height?: number;
  showPredicted?: boolean;
  valueFormatter?: (v: number) => string;
}

export default function TrendLineChart({
  data,
  height = 280,
  showPredicted = false,
  valueFormatter = (v) => v.toLocaleString(),
}: TrendLineChartProps) {
  return (
    <ResponsiveContainer width="100%" height={height}>
      <LineChart data={data} margin={{ top: 5, right: 20, bottom: 5, left: 0 }}>
        <CartesianGrid strokeDasharray="3 3" stroke="#2a2b36" />
        <XAxis
          dataKey="label"
          tick={{ fill: "#9394a1", fontSize: 11 }}
          tickLine={false}
          axisLine={{ stroke: "#2a2b36" }}
        />
        <YAxis
          tick={{ fill: "#9394a1", fontSize: 11 }}
          tickLine={false}
          axisLine={false}
          tickFormatter={valueFormatter}
        />
        <Tooltip
          contentStyle={{
            background: "#12131a",
            border: "1px solid #2a2b36",
            borderRadius: 8,
            fontSize: 12,
          }}
          labelStyle={{ color: "#f0f0f3" }}
          formatter={(value: number) => [valueFormatter(value), ""]}
        />
        <Line
          type="monotone"
          dataKey="value"
          stroke="#6366f1"
          strokeWidth={2}
          dot={false}
          activeDot={{ r: 4, fill: "#6366f1" }}
        />
        {showPredicted && (
          <Line
            type="monotone"
            dataKey="predicted"
            stroke="#818cf8"
            strokeWidth={2}
            strokeDasharray="5 5"
            dot={false}
          />
        )}
      </LineChart>
    </ResponsiveContainer>
  );
}
