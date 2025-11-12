import React, { useMemo } from 'react'
import { motion } from 'framer-motion'

export function TextShimmer({
  children,
  as: Component = 'p',
  className = '',
  duration = 2,
  spread = 2,
  color = '#ffffff',
  gradientColor = '#8b5cf6'
}) {
  const MotionComponent = motion(Component)

  const dynamicSpread = useMemo(() => {
    return children.length * spread
  }, [children, spread])

  return (
    <MotionComponent
      className={`relative inline-block bg-[length:250%_100%,auto] bg-clip-text text-transparent ${className}`}
      style={{
        '--spread': `${dynamicSpread}px`,
        backgroundImage: `linear-gradient(90deg, transparent calc(50% - var(--spread)), ${gradientColor}, transparent calc(50% + var(--spread))), linear-gradient(${color}, ${color})`,
        WebkitBackgroundClip: 'text',
        WebkitTextFillColor: 'transparent',
      }}
      initial={{ backgroundPosition: '100% center' }}
      animate={{ backgroundPosition: '0% center' }}
      transition={{
        repeat: Infinity,
        duration,
        ease: 'linear',
      }}
    >
      {children}
    </MotionComponent>
  )
}

