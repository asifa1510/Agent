import React from 'react';
import { NavLink } from 'react-router-dom';

const navigation = [
  { name: 'Dashboard', href: '/', icon: '📊' },
  { name: 'Sentiment', href: '/sentiment', icon: '💭' },
  { name: 'Predictions', href: '/predictions', icon: '📈' },
  { name: 'Trades', href: '/trades', icon: '💰' },
  { name: 'Portfolio', href: '/portfolio', icon: '📋' },
];

export function Navigation() {
  return (
    <nav className="bg-white shadow-sm border-b border-gray-200">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex space-x-8">
          {navigation.map((item) => (
            <NavLink
              key={item.name}
              to={item.href}
              className={({ isActive }) =>
                `inline-flex items-center px-1 pt-1 pb-4 border-b-2 text-sm font-medium transition-colors ${
                  isActive
                    ? 'border-blue-500 text-blue-600'
                    : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
                }`
              }
            >
              <span className="mr-2">{item.icon}</span>
              {item.name}
            </NavLink>
          ))}
        </div>
      </div>
    </nav>
  );
}

export default Navigation;