/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        ivory: '#FAF5EF',
        coral: '#6D263D',
        amber: '#D6AD70',
        slatecobalt: '#C97861',
        rose: '#6D263D',
        midnight: '#3B292B',
        subtleborder: '#E9DCD2',
      },
      fontFamily: {
        serif: ['"Cormorant Garamond"', 'Georgia', 'serif'],
        sans: ['"Plus Jakarta Sans"', 'system-ui', 'sans-serif'],
        mono: ['"JetBrains Mono"', 'monospace'],
      },
      boxShadow: {
        editorial: '0 24px 60px -15px rgba(59, 41, 43, 0.14)',
        editorialLg: '0 32px 80px -20px rgba(59, 41, 43, 0.22)',
        pill: '0 10px 30px -8px rgba(59, 41, 43, 0.15)',
      },
    },
  },
  plugins: [],
};
