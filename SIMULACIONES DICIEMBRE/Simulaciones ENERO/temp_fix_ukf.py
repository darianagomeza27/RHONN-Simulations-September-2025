#!/usr/bin/env python3
"""
Script temporal para arreglar las llamadas a UKF_Trainer
"""

# Leer archivo
with open('simulaciones100126.py', 'r') as f:
    lines = f.readlines()

# Reemplazar línea 339 (índice 338)
if 'trainer = UKF_Trainer(n_states, eta=eta, alpha=alpha)' in lines[338]:
    lines[338] = '            trainer = UKF_Trainer(n_states, eta=eta, alpha=alpha, Q=1e-4, R=1e-5)\n'
    print("✓ Línea 339 modificada")
else:
    print("✗ Línea 339 no coincide")

# Reemplazar línea 1310 (índice 1309)
if 'trainer = UKF_Trainer(n_states, eta=eta, alpha=alpha)' in lines[1309]:
    lines[1309] = '            trainer = UKF_Trainer(n_states, eta=eta, alpha=alpha, Q=1e-4, R=1e-5)\n'
    print("✓ Línea 1310 modificada")
else:
    print("✗ Línea 1310 no coincide")

# Escribir archivo
with open('simulaciones100126.py', 'w') as f:
    f.writelines(lines)

print("\n✅ Archivo actualizado")
