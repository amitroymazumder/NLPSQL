// frontend/pages/index.tsx

import { useState } from 'react';
import axios from 'axios';
import {
  LineChart, BarChart, PieChart, AreaChart, XAxis, YAxis, Tooltip, CartesianGrid, Legend, Line, Bar, Pie, Area, Cell, ResponsiveContainer
} from 'recharts';

const ChartTypes: any = {
  bar: BarChart,
  line: LineChart,
  area: AreaChart,
  pie: PieChart
};

export default function Home() {
  const [query, setQuery] = useState('');
  const [sql, setSql] = useState('');
  const [data, setData] = useState<any[]>([]);
  const [chartConfig, setChartConfig] = useState<any>(null);
  const [summary, setSummary] = useState('');
  const [view, setView] = useState<'chart' | 'table'>('chart');
  const [loading, setLoading] = useState(false);

  const fetchData = async () => {
    setLoading(true);
    try {
      const res = await axios.post('/api/nl-to-sql', { query });
      setSql(res.data.sql);
      setData(res.data.data);
      setChartConfig(res.data.chartConfig);
      setSummary(res.data.summary);
      setView('chart');
    } catch (err) {
      console.error('Error fetching data:', err);
    } finally {
      setLoading(false);
    }
  };

  const renderChart = () => {
    if (!chartConfig || !data.length) return null;
    const ChartComponent = ChartTypes[chartConfig.chartType || 'bar'];

    return (
      <ResponsiveContainer width="100%" height={400}>
        <ChartComponent data={data}>
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis dataKey={chartConfig.xAxis} />
          <YAxis />
          <Tooltip />
          <Legend />
          {(chartConfig.chartType === 'bar') && <Bar dataKey={chartConfig.yAxis} fill="#8884d8" />}
          {(chartConfig.chartType === 'line') && <Line dataKey={chartConfig.yAxis} stroke="#8884d8" />}
          {(chartConfig.chartType === 'area') && <Area dataKey={chartConfig.yAxis} stroke="#8884d8" fill="#82ca9d" />}
        </ChartComponent>
      </ResponsiveContainer>
    );
  };

  const renderTable = () => (
    <div className="overflow-x-auto">
      <table className="table-auto w-full border border-gray-300">
        <thead>
          <tr>
            {data[0] && Object.keys(data[0]).map(key => (
              <th key={key} className="border px-4 py-2">{key}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {data.map((row, i) => (
            <tr key={i}>
              {Object.values(row).map((val, j) => (
                <td key={j} className="border px-4 py-2">{val}</td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );

  return (
    <main className="p-6 max-w-4xl mx-auto">
      <h1 className="text-2xl font-bold mb-4">Natural Language SQL + Chart UI</h1>

      <textarea
        value={query}
        onChange={(e) => setQuery(e.target.value)}
        className="w-full p-2 border rounded mb-2"
        placeholder="Ask something like: Total notional by product last month"
      />
      <button
        onClick={fetchData}
        disabled={loading}
        className="bg-blue-600 text-white px-4 py-2 rounded hover:bg-blue-700"
      >
        {loading ? 'Loading...' : 'Run Query'}
      </button>

      {sql && (
        <div className="mt-6 text-sm bg-gray-100 p-3 rounded border">
          <strong>Generated SQL:</strong>
          <pre className="whitespace-pre-wrap break-words mt-1">{sql}</pre>
        </div>
      )}

      {data.length > 0 && (
        <div className="mt-4">
          <div className="flex space-x-4 mb-2">
            <button onClick={() => setView('chart')} className={`px-4 py-1 rounded ${view === 'chart' ? 'bg-blue-600 text-white' : 'bg-gray-200'}`}>Chart</button>
            <button onClick={() => setView('table')} className={`px-4 py-1 rounded ${view === 'table' ? 'bg-blue-600 text-white' : 'bg-gray-200'}`}>Table</button>
          </div>

          {view === 'chart' ? renderChart() : renderTable()}
        </div>
      )}

      {summary && (
        <div className="mt-6 p-4 bg-yellow-50 border border-yellow-300 rounded">
          <strong>Summary:</strong>
          <p className="mt-1 text-gray-800">{summary}</p>
        </div>
      )}
    </main>
  );
}
