"""Explicit semantic categories and fixed styles for this analysis only."""
COMMON_STYLES = {
    'Input setup': {'color': '#999999', 'hatch': '...'},
    'Block normalization': {'color': '#56B4E9', 'hatch': ''},
    'QKV projection': {'color': '#0072B2', 'hatch': ''},
    'RoPE / context layout': {'color': '#E69F00', 'hatch': '///'},
    'Attention': {'color': '#F0E442', 'hatch': ''},
    'MLP up projection': {'color': '#009E73', 'hatch': '///'},
    'H/Z output block': {'color': '#CC79A7', 'hatch': ''},
    'Final normalization': {'color': '#000000', 'hatch': '///'},
    'Vocabulary projection': {'color': '#D55E00', 'hatch': ''},
}
CATEGORY_TO_COMMON = {
    'Position indices': 'Input setup', 'Token embedding': 'Input setup',
    'RoPE tables': 'Input setup', 'Block layer norms': 'Block normalization',
    'QKV projection': 'QKV projection', 'RoPE / QKV layout': 'RoPE / context layout',
    'Context layout': 'RoPE / context layout', 'Attention': 'Attention',
    'MLP up projection': 'MLP up projection', 'H gate': 'H/Z output block',
    'Z gate': 'H/Z output block',
    'Z down projection (gate fused if applicable)': 'H/Z output block',
    'H down projection': 'H/Z output block', 'H feature compaction': 'H/Z output block',
    'Joint H/Z gates / down / residual': 'H/Z output block',
    'Residual sum': 'H/Z output block', 'Final norm': 'Final normalization',
    'Vocabulary projection': 'Vocabulary projection',
}
DETAIL_MAP = {
    'H gate': 'H gate / preparation', 'H feature compaction': 'H gate / preparation',
    'H down projection': 'H down (W2)',
    'Z gate': 'Z gate + down (Wo)',
    'Z down projection (gate fused if applicable)': 'Z gate + down (Wo)',
    'Residual sum': 'Residual sum',
    'Joint H/Z gates / down / residual': 'Fused H/Z output block (indivisible)',
}
DETAIL_HATCHES = {'H gate / preparation': '...', 'H down (W2)': '///',
                  'Z gate + down (Wo)': 'xxx', 'Residual sum': 'ooo',
                  'Fused H/Z output block (indivisible)': ''}
MODE_STYLES = {
    'native': {'color': '#0072B2', 'marker': 'o', 'linestyle': '-'},
    'generic_sparse': {'color': '#777777', 'marker': 's', 'linestyle': '--'},
    'generic_no_skip': {'color': '#777777', 'marker': 'x', 'linestyle': ':'},
    'tile_sparse': {'color': '#009E73', 'marker': '^', 'linestyle': '--'},
    'sparse': {'color': '#009E73', 'marker': '^', 'linestyle': '--'},
    'tile_no_skip': {'color': '#D55E00', 'marker': 'D', 'linestyle': ':'},
    'no_skip': {'color': '#D55E00', 'marker': 'D', 'linestyle': ':'},
}
DEFINITIONS = {
    'Input setup': 'Índices de posição, embedding de tokens e tabelas de RoPE, fora dos seis blocos.',
    'Block normalization': 'LayerNorm dos ramos de atenção e MLP, separadas ou em um kernel pareado; exclui a norma final.',
    'QKV projection': 'Projeção que produz query, key e value.',
    'RoPE / context layout': 'Rotação RoPE e rearranjos/empacotamento de QKV e contexto; exclui a geração inicial das tabelas.',
    'Attention': 'QK causal, softmax e PV, incluindo atividades split/combine; exclui o gate z e a projeção Wo.',
    'MLP up projection': 'Projeção de expansão W1; exclui o gate h e a projeção W2.',
    'H/Z output block': 'Gates h/z + preparação para o consumidor esparso + W2 (h) + Wo (z) + combinação com o residual; inclui custos de prepass, compactação, fallback, bias e arredondamento quando embutidos nas etapas medidas.',
    'Final normalization': 'LayerNorm após o último bloco e antes da projeção de vocabulário.',
    'Vocabulary projection': 'Projeção para os 50.304 logits de cada um dos 2.048 tokens.',
}


def common_values(categories):
    values = dict.fromkeys(COMMON_STYLES, 0.)
    for category, value in categories.items():
        values[CATEGORY_TO_COMMON[category]] += value
    return values


def detail_values(categories):
    values = {}
    for category, value in categories.items():
        if category in DETAIL_MAP:
            label = DETAIL_MAP[category]
            values[label] = values.get(label, 0.) + value
    return values


def dictionary_text():
    rows = ['| Categoria comum (label do gráfico) | Cor fixa | Definição |', '| --- | --- | --- |']
    rows += [f'| {label} | {style["color"]} | {DEFINITIONS[label]} |' for label, style in COMMON_STYLES.items()]
    mapping = ['| Etapa disponível na medição original | Categoria comum |', '| --- | --- |']
    mapping += [f'| {source} | {target} |' for source, target in CATEGORY_TO_COMMON.items()]
    return '\n'.join([
        '# Definições e cores da decomposição', '',
        '## Comparação entre modelos', '',
        'As figuras 02, 03, 04 e 08 e as tabelas principais usam nove categorias com os mesmos labels, cores, padrões e ordem, para todos os tamanhos, thresholds e modos de execução. As cores identificam o trabalho realizado; o título informa se o threshold é tiled ou scalar. A figura 08 reúne os três tamanhos em pares dense/sparse.', '',
        *rows, '',
        '**H/Z output block é o agrupamento comparável.** No 14M/31M otimizado, ele é medido pelo consumidor fundido. No 70M e nas referências nativas, é a soma das etapas correspondentes disponíveis. Ele inclui mais que as duas projeções: também gates, preparação esparsa e soma residual.', '',
        'As barras são um orçamento de atividades CUDA, não uma linha do tempo. Um agrupamento pode somar etapas separadas na sequência de execução. Não é um novo kernel nem um intervalo contínuo medido. Os valores originais e o total de atividades são preservados; a apresentação só agrega categorias.', '',
        '**Fusão limita o detalhamento disponível.** Não há tempos independentes de h gate, h down, z gate, z down e residual dentro do consumidor fundido de 14M/31M. Não dividimos seu tempo por FLOPs, esparsidade ou proporções do 70M. Em uma tabela detalhada, N/A significa que a etapa não é medida separadamente, e não custo zero.', '',
        '## Mapeamento das etapas originais', '', *mapping, '',
        '## Detalhamento do H/Z output block', '',
        'As figuras 05, 06 e 07 mostram somente esse bloco. Todos os seus componentes ficam na mesma família roxa (#CC79A7); os padrões identificam componentes medidos. O mapeamento é fixo:', '',
        '| Label do detalhe | Etapas somadas | Padrão |', '| --- | --- | --- |',
        '| H gate / preparation | H gate e H feature compaction, quando separados | Pontos |',
        '| H down (W2) | H down projection, densa ou gathered | Diagonal |',
        '| Z gate + down (Wo) | Z gate e Z down; inclui o gate quando já fundido à projeção | Cruzado |',
        '| Residual sum | Soma residual separada | Círculos |',
        '| Fused H/Z output block (indivisible) | Consumidor conjunto completo, sem separação interna | Sólido |', '',
        'No 70M, a parcela H down pode ser dominante com skipping desabilitado. Ela é um componente do mesmo bloco roxo que aparece inteiro no 31M; não é uma categoria concorrente a ele. O z do 70M continua com aritmética densa.', '',
        'Nos gráficos de linhas, a cor identifica o modo de execução: azul para native; verde para o sparse selecionado; vermelho para o no-skip selecionado; cinza para o generic adicional do 14M. Isso é outra legenda, explicitamente indicada. Os mesmos modos mantêm a mesma cor e marcador entre tamanhos.', '',
        '## Regras de threshold, separadas da legenda de operações', '',
        '**14M: tiled threshold 8×16**, com decisão pelo canto superior esquerdo. **31M/70M: scalar threshold elemento a elemento.** Grupos de tokens, compactação e tiles usados no produto matricial não tornam o threshold tiled. Native usa aritmética densa no mesmo checkpoint com gate; no-skip mantém o gate e desativa o skipping do consumidor.', '',
        'Unidades: médias aritméticas de durações CUPTI em microssegundos para a decomposição; a latência host completa usa média geométrica independente. Não há reescala de componentes para fechar artificialmente a latência host.', '',
        '[Relatório e figuras](README.md) · [Mapeamento verificável](results/category-mapping.csv) · [Categorias comuns](results/semantic-categories.csv) · [Etapas originais](results/forward-stages.csv)',
    ]) + '\n'
