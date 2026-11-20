# RUN-042 blind review

Cambios: operación (snapshot/kill/relaunch), SESSION_HANDOFF append,
scripts/_constr_decode.py (dtype fix ya cubierto en RUN-041).

- Snapshot antes de mutar ckpt: cumple la regla de checkpoint (RUN-039).
- Kill -9 justificado: save atómico por época → estado en disco = ep13.
- Relaunch sin COMPILE: decisión correcta — dynamo falla sin g++ y la
  medición previa mostró empate de velocidad. El comportamiento efectivo
  del run anterior también era eager (compile fallaba silenciosamente).
- Riesgo menor: holdout movido a 19000:20000 — cualquier eval futura con
  OFFSET=8000 estaría midiendo en datos de entrenamiento; documentado en
  handoff. Aceptable.
- Riesgo menor: train_code4.log depende de `podman logs -f` en background —
  si el proceso de forwarding muere, el log se detiene aunque el trainer
  siga. Mitigable re-adjuntando logs. No bloquea.
- Sin secretos, sin red, mounts correctos.

Veredicto: sin bloqueadores.
