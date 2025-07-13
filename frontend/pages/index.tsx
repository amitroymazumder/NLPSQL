import { useState } from 'react';
import axios from 'axios';
import { BarChart, Bar, XAxis, YAxis, Tooltip, CartesianGrid, ResponsiveContainer } from 'recharts';

export default function HomePage() {
  const [query, setQuery] = useState('');
  const [data, setData] = useState([]);
  const [chartConfig, setChartConfig] = useState(null);
  const [loading, setLoading] = useState(false);

  const submitQuery = async () => {
    setLoading(true);
    try {
      const response = await axios.post('/api/nl-to-sql', { query });
      setData(response.data.data);
      setChartConfig(response.data.chartConfig);
    } catch (err) {
      alert('Failed to fetch results');
    }
    setLoading(false);
  };

  return (
    <div className="p-8 max-w-4xl mx-auto">
      <h1 className="text-2xl font-bold mb-4">Natural Language to SQL + Chart</h1>
      <textarea
        className="w-full p-2 border rounded mb-4"
        rows={3}
        value={query}
        onChange={(e) => setQuery(e.target.value)}
        placeholder="Ask a question about trades..."
      />
      <button className="bg-blue-600 text-white px-4 py-2 rounded" onClick={submitQuery} disabled={loading}>
        {loading ? 'Loading...' : 'Run Query'}
      </button>

      {chartConfig && (
        <div className="mt-8">
          <h2 className="text-xl font-semibold mb-2">{chartConfig.title}</h2>
          <ResponsiveContainer width="100%" height={400}>
            <BarChart data={data}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey={chartConfig.xAxis} />
              <YAxis />
              <Tooltip />
              <Bar dataKey={chartConfig.yAxis} fill="#8884d8" />
            </BarChart>
          </ResponsiveContainer>
        </div>
      )}
    </div>
  );
}
