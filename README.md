# FLEWS - Flood Early Warning System

A real-time flood monitoring and early warning application with an interactive Mapbox map interface.

## Features

- 🗺️ Interactive Mapbox map centered at specified coordinates (34°4'9.606"N, 72°38'36.1464"E)
- 🎨 Beautiful UI with Tailwind CSS and SCSS
- ⚡ Built with React, TypeScript, and Vite
- 🎭 Smooth animations using Framer Motion
- 📱 Fully responsive design
- 🌊 Real-time flood monitoring and alerts

## Prerequisites

- Node.js (v18 or higher recommended)
- npm or yarn
- A Mapbox account and access token

## Getting Started

### 1. Clone the repository

```bash
git clone <your-repo-url>
cd FLEWS
```

### 2. Install dependencies

```bash
npm install
```

### 3. Set up Mapbox Access Token

1. Go to [Mapbox](https://account.mapbox.com/access-tokens/) and create a free account
2. Create a new access token or use your default public token
3. Create a `.env` file in the root directory:

```bash
cp .env.example .env
```

4. Add your Mapbox token to the `.env` file:

```
VITE_MAPBOX_TOKEN=your_actual_mapbox_token_here
```

### 4. Run the development server

```bash
npm run dev
```

The application will be available at `http://localhost:5173` (or another port if 5173 is in use).

## Available Scripts

- `npm run dev` - Start development server
- `npm run build` - Build for production
- `npm run preview` - Preview production build
- `npm run lint` - Run ESLint

## Technologies Used

- **React 19** - UI library
- **TypeScript** - Type safety
- **Vite** - Build tool and dev server
- **Mapbox GL JS** - Interactive maps
- **Tailwind CSS v4** - Utility-first CSS framework
- **@tailwindcss/postcss** - Tailwind PostCSS plugin
- **SCSS** - CSS preprocessor
- **Framer Motion** - Animation library
- **Chakra UI** - Component library
- **Zustand** - State management

## Customization

### Changing Map Center

To change the map center coordinates, edit `src/components/MapBackground.tsx`:

```typescript
const lat = 34.069335;  // Your latitude
const lng = 72.643374;  // Your longitude
```

### Changing Map Style

You can change the map style in `src/components/MapBackground.tsx`:

```typescript
style: 'mapbox://styles/mapbox/dark-v11',
```

Available styles:
- `mapbox://styles/mapbox/streets-v12`
- `mapbox://styles/mapbox/outdoors-v12`
- `mapbox://styles/mapbox/light-v11`
- `mapbox://styles/mapbox/dark-v11`
- `mapbox://styles/mapbox/satellite-v9`
- `mapbox://styles/mapbox/satellite-streets-v12`
- `mapbox://styles/mapbox/navigation-day-v1`
- `mapbox://styles/mapbox/navigation-night-v1`

## License

This project is free and open source (FOSS).

## Contributors

Hamza Bin Aamir, Ahmed Abdullah, Syed Areeb Zaheer, Azeem Liaqat
