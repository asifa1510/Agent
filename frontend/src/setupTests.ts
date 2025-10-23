// jest-dom adds custom jest matchers for asserting on DOM nodes.
// allows you to do things like:
// expect(element).toHaveTextContent(/react/i)
// learn more: https://github.com/testing-library/jest-dom
import '@testing-library/jest-dom';

// Mock Chart.js and related modules for testing
jest.mock('chart.js', () => ({
  Chart: {
    register: jest.fn(),
  },
  CategoryScale: {},
  LinearScale: {},
  PointElement: {},
  LineElement: {},
  Title: {},
  Tooltip: {},
  Legend: {},
  TimeScale: {},
  Filler: {},
}));

jest.mock('react-chartjs-2', () => ({
  Line: jest.fn(() => null),
}));

jest.mock('chartjs-adapter-date-fns', () => ({}));