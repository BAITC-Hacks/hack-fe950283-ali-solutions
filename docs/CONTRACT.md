# Контракт MoneyGraph 1.0

## Данные и CSV

Вход: nodes(gid:int64, depth:0..4, is_seed:bool); edges(src:int64, dst:int64, sum_kzt>0, n_tx:positive integer, depth:1..4); transactions(src:int64, dst:int64, date:calendar date, sum_kzt>0).
Пустой набор nodes отклоняется; узлы без рёбер поддерживаются. Дубли транзакций сохраняются, поскольку transaction_id отсутствует.

Обязательные колонки не изменены:

- nodes_roles.csv: gid, role, role_score, cluster_id, priority_score, evidence.
- clusters.csv: cluster_id, n_nodes, n_seed, sum_kzt_internal, top_gids, hypothesis.
- top_nodes.csv: rank, gid, role, priority_score, why.

gid не преобразуется через float. В CSV он записывается десятичным int64, в JSON/UI всегда строкой. top_gids в CSV разделён точкой с запятой, в JSON — список строк.
Evidence содержит число и не превышает 200 символов. why не обрезается. Top содержит min(50,N) строк, сортировка priority DESC, gid ASC. Дополнительные неизвестные признаки в JSON — null, в CSV — пустое поле.

## Запросы

POST /api/query принимает JSON с action, args и необязательным run_id. Неверные аргументы дают HTTP 400, несогласованный run_id — 409.

- get_node_report(gid:str): показатели, правила, компоненты приоритета, время, до 20 входящих/исходящих рёбер, Markdown.
- find_common_recipients(gids:list[str], mode="direct", max_hops=1): 2..20 входных значений, удаление дубликатов, минимум два разных известных узла; совпадение всех источников. direct требует max_hops=1; reachable допускает 1..4. До 50 результатов, отсортированных по приоритету.
- trace_paths_from_seeds(gid:str, max_hops=4, limit=10): до 50 путей; один кратчайший на каждый достижимый seed; нулевой путь к себе исключён. Направление и ограничения явные.
- get_cluster_report(cluster_id:int): размер, состав, оборот и гипотеза.

Ответ: schema_version, run_id, data_hash, result, facts, evidence_refs, limitations, truncated, limits, highlight_gids.
Каждый факт содержит серверный id, kind и value. Идентификатор связан с run и содержимым результата. Суммы последовательных рёбер не складываются в доставленную сумму.

Существующие /api/ask и /api/health сохранены. Legacy tools доступны через адаптер GraphTools, используют тот же рассчитанный контекст.
Скачивание через локальный сервер: /download/{nodes_roles,clusters,top_nodes}.csv и /api/node-report/{gid}.md; Content-Disposition=attachment.

## Воспроизводимость и публикация

run_id определяется хешами входа, конфигурации, исполняемого кода и версиями зависимостей. Содержимое результата связано с run; поля времени измерения могут отличаться между запусками.
Порядок gid/рёбер фиксирован перед алгоритмами, random seed=42. Результат гарантируется в проверенном окружении, не между любыми версиями NetworkX.

Комплект записывается во временный соседний каталог, проверяется, затем заменяет предыдущий каталог out. Lock запрещает параллельных писателей. При ошибке до публикации прежний результат сохраняется. При замене каталога возможен короткий интервал недоступности пути; смешанный комплект не публикуется.
Если процесс аварийно завершён во время записи, возможен оставшийся lock. Удалять его следует только после проверки, что писатель действительно остановлен.
