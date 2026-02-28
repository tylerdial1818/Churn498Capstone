import {
  ResponsiveContainer,
  BarChart as RechartsBarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
} from "recharts";

interface BarChartProps {
  data: { label: string; value: number }[];
  height?: number;
  color?: string;
  horizontal?: boolean;
  valueFormatter?: (v: number) => string;
}

export default function BarChart({
  data,
  height = 280,
  color = "#6366f1",
  horizontal = false,
  valueFormatter = (v) => v.toLocaleString(),
}: BarChartProps) {
  if (horizontal) {
    return (
      <ResponsiveContainer width="100%" height={height}>
        <RechartsBarChart data={data} layout="vertical" margin={{ top: 5, right: 20, bottom: 5, left: 80 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#2a2b36" horizontal={false} />
          <XAxis
            type="number"
            tick={{ fill: "#9394a1", fontSize: 11 }}
            tickLine={false}
            axisLine={{ stroke: "#2a2b36" }}
            tickFormatter={valueFormatter}
          />
          <YAxis
            type="category"
            dataKey="label"
            tick={{ fill: "#9394a1", fontSize: 11 }}
            tickLine={false}
            axisLine={false}
            width={75}
          />
          <Tooltip
            contentStyle={{
              background: "#12131a",
              border: "1px solid #2a2b36",
              borderRadius: 8,
              fontSize: 12,
            }}
            formatter={(value?: number) => [valueFormatter(value ?? 0), ""]}
          />
          <Bar dataKey="value" fill={color} radius={[0, 4, 4, 0]} />
        </RechartsBarChart>
      </ResponsiveContainer>
    );
  }

  return (
    <ResponsiveContainer width="100%" height={height}>
      <RechartsBarChart data={data} margin={{ top: 5, right: 20, bottom: 5, left: 0 }}>
        <CartesianGrid strokeDasharray="3 3" stroke="#2a2b36" vertical={false} />
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
          formatter={(value?: number) => [valueFormatter(value ?? 0), ""]}
        />
        <Bar dataKey="value" fill={color} radius={[4, 4, 0, 0]} />
      </RechartsBarChart>
    </ResponsiveContainer>
  );
}
