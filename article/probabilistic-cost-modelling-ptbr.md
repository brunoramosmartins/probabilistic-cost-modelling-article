# A Forma do Que Você Vai Gastar

## Modelagem Probabilística de Custos de Pessoal — da Seleção de Distribuições ao Impacto Orçamentário

---

## Sumário Executivo

Este artigo demonstra que a distribuição Normal — assumida implicitamente na maioria dos modelos de custo — subestima sistematicamente o risco de cauda em projeções de custos de pessoal. Usando Estimação por Máxima Verossimilhança, critérios informacionais (AIC/BIC) e testes de aderência, derivamos um framework principiado para seleção de distribuições. O resultado central: assumir Normal em vez de Pareto para custos de rescisão subestima a probabilidade de despesas extremas por um fator de **138x**, com implicações diretas sobre reservas orçamentárias.

---

## O Que É Este Artigo

Este é um **framework estatístico prático** para seleção de distribuições em modelagem de custos de pessoal. Não é um livro-texto de probabilidade nem um tutorial de Python. Assume que você aceita que "distribuição errada = orçamento errado" e quer uma forma rigorosa e reprodutível de escolher a certa.

O artigo vai da teoria (Seções 3–7) aos experimentos (Seção 8) e a um framework de decisão em cinco passos (Seção 9). O repositório associado contém todo o código, geradores de dados sintéticos e figuras reprodutíveis.

---

## O Que Você Precisa Saber

Este artigo assume familiaridade com:

**Necessário:**
- Cálculo: derivadas, integrais, expansão de Taylor básica
- Probabilidade básica: PDF, CDF, esperança, variância
- Álgebra linear: inversão de matrizes (usada brevemente para informação de Fisher)

**Útil mas não obrigatório:**
- Exposição prévia à Estimação por Máxima Verossimilhança (derivamos do zero)
- Familiaridade com critérios informacionais (AIC/BIC são derivados na Seção 5)

**Fora de escopo (não cobrimos):**
- Inferência Bayesiana e MCMC
- Modelagem de séries temporais e dependência (mencionado brevemente em Limitações)
- Coleta de dados reais / NDA / questões de privacidade

---

## Notação

| Símbolo | Significado |
|---------|-------------|
| $X$ | Variável aleatória (componente de custo) |
| $f(x \mid \theta)$ | Função densidade de probabilidade com parâmetro $\theta$ |
| $F(x)$ | Função de distribuição acumulada |
| $\hat{\theta}$ | Estimador de Máxima Verossimilhança de $\theta$ |
| $\ell(\theta)$ | Função log-verossimilhança |
| $S(\theta)$ | Função score (gradiente de $\ell$) |
| $I(\theta)$ | Informação de Fisher |
| $n$ | Tamanho amostral |
| $k$ | Número de parâmetros estimados |
| $D_{KL}(p \| q)$ | Divergência de Kullback-Leibler de $p$ para $q$ |
| $\text{VaR}_p$ | Value at Risk no nível de confiança $p$ |
| $\text{ES}_p$ | Expected Shortfall no nível $p$ |

---

## 1. Introdução: Por Que Distribuições Importam

Um orçamento é uma declaração probabilística disfarçada de planilha. Quando um analista projeta custos de pessoal, está implicitamente assumindo uma distribuição de probabilidade para cada componente — salários, horas extras, rescisões, contratações. Na maioria das organizações, essa suposição é a distribuição Normal: simétrica, com caudas leves, bem-comportada.

**Essa suposição pode quebrar o orçamento de um trimestre.** Sob uma distribuição Normal, a probabilidade de um evento de rescisão exceder R\$ 50.000 é cerca de 0,013% — uma vez a cada 77.000 casos. Sob a distribuição Pareto que de fato modela esse tipo de dado, a mesma probabilidade é 1,79% — uma vez a cada 56 casos. **Uma diferença de 138x.** Quando essa diferença aparece em rescisões reais, a reserva de caixa calibrada para a Normal deixa de ser margem de segurança e vira ficção.

O problema é que custos de pessoal não são Normais. Salários são assimétricos à direita: a maioria dos colaboradores recebe salários moderados, enquanto poucos executivos puxam a média para cima. Custos de rescisão são de cauda pesada: a maioria é moderada, mas alguns poucos casos extremos dominam o total. Dados salariais frequentemente são multimodais: clusters de juniores, plenos e seniores formam picos distintos que nenhuma distribuição unimodal pode capturar.

**A distribuição que você assume é o seu modelo.** Todo o resto — média, variância, intervalos de confiança, estimativas de risco — flui dessa escolha. Uma distribuição errada não é uma nuance de modelagem; é um viés sistemático que se propaga por todo cálculo a jusante.

$$\text{Distribuição errada} \rightarrow \text{parâmetros errados} \rightarrow \text{orçamento errado} \rightarrow \text{decisões erradas}$$

Este artigo apresenta um framework rigoroso para seleção de distribuições em modelagem de custos. Derivamos a Estimação por Máxima Verossimilhança (MLE) a partir dos fundamentos, ajustamos cinco famílias de distribuições candidatas a dados sintéticos de custos, e usamos critérios informacionais (AIC, BIC) e testes de aderência para selecionar o melhor modelo. O artigo complementar (Monte Carlo) mostra como usar essas distribuições para simular o custo total de uma equipe.

---

## 2. O Que Está em Jogo

Antes de mergulhar no formalismo, três números enquadram por que a escolha importa. A figura abaixo prevê a comparação central: sob um modelo Normal, a reserva orçamentária parece confortável — até a realidade seguir uma Pareto.

![O que está em jogo: reserva orçamentária sob suposição correta vs errada](../figures/wrong_distribution_impact.png)

**Três impactos concretos em uma equipe de 50 pessoas:**

1. **Gap de probabilidade de cauda (rescisão):** $P(X > 50{.}000)$ — ou seja, a probabilidade de uma rescisão exceder R\$ 50.000 — é **138x maior** sob Pareto que sob Normal. O "evento raro" que a Normal prevê torna-se ocorrência rotineira sob o modelo correto.

2. **Bimodalidade oculta (salário):** uma única Normal ajustada a uma mistura de juniores/seniores infla a estimativa de variância em ~40% enquanto a média não representa nenhum colaborador real. O VaR a 95% acaba errado em ambas as direções.

3. **Subprovisão de reserva (orçamento anual):** para uma equipe de 50 pessoas onde salários são LogNormal e rescisões são Pareto, assumir Normal subestima a reserva no VaR 99% em **R\$ 100.000–R\$ 150.000 por ano**. É a diferença entre "temos margem" e "estamos expostos".

O resto do artigo mostra como detectar, quantificar e corrigir cada um desses gaps.

---

## 3. Os Componentes de Custo

Para tornar o framework concreto, precisamos de um modelo do que estamos estimando. Representamos cada componente de custo como uma variável aleatória com propriedades distribucionais distintas. O modelo é genérico, mínimo e expansível.

| Componente | Símbolo | Candidatas | Justificativa |
|------------|---------|------------|---------------|
| Salário base | $S_i$ | LogNormal, Gamma, Mistura(Normal) | Assimétrico à direita; multimodal se há clusters |
| Custo de hora extra | $C_{ot}$ | LogNormal, Gamma | Assimétrico, sempre positivo |
| Custo de rescisão | $C_{sev}$ | **Pareto**, LogNormal | Cauda pesada: poucos casos extremamente caros |
| Custo de contratação | $C_h$ | LogNormal, Gamma | Variável (fees de recrutador, relocação) |
| Multiplicador de benefícios | $\beta_i$ | Uniform, Beta | Limitado: tipicamente 1,3x–2,2x do salário |

### O Baseline da Planilha

Antes de introduzir modelos melhores, vale nomear aquele sendo substituído. A planilha típica de FP&A trata cada componente de custo como $\bar{x} \pm k \cdot s$ — uma média e um desvio-padrão, frequentemente com $k = 2$ ou $k = 3$. Isso é implicitamente um modelo Normal: assume comportamento simétrico e de cauda leve em torno da média. Para salários, subestima a cauda direita; para rescisões, subestima catastroficamente. O framework abaixo é uma substituição estruturada para esse baseline implícito.

### Exemplo Concreto: Equipe de 50 Pessoas

Considere uma equipe de TI com 50 colaboradores. Usando parâmetros calibrados para o mercado brasileiro:

- **Salários**: LogNormal($\mu = 9.1$, $\sigma = 0.4$), mediana ~ R\$ 9.000/mês
- **Horas extras**: Gamma($\alpha = 4$, $\beta = 1/30$), média ~ R\$ 120/hora
- **Rescisões**: Pareto($\alpha = 2.5$, $x_m = 10.000$), ~3 eventos/ano
- **Contratações**: LogNormal($\mu = 9.5$, $\sigma = 0.7$), ~5 contratações/ano

O custo anual total esperado é de aproximadamente R\$ 6,0–6,5 milhões. A questão crucial: a *variância* desse total depende criticamente de quais distribuições você assume. Suposições Normais produzem um intervalo de confiança estreito; suposições corretas de cauda pesada produzem um intervalo muito mais amplo.

---

## 4. Famílias de Distribuições

Agora que sabemos quais componentes precisamos modelar, precisamos de um vocabulário de distribuições candidatas que possam capturar suas formas distintas. Para cada componente, consideramos cinco famílias. A Normal entra como o baseline a ser superado — não como candidata séria.

### As Cinco Candidatas

**Normal** $N(\mu, \sigma^2)$: A suposição padrão — e frequentemente errada. Simétrica, caudas leves, suporte em $(-\infty, \infty)$. Atribui probabilidade positiva a custos negativos, o que é fisicamente impossível para salários.

**LogNormal** $\text{LogNormal}(\mu, \sigma^2)$: Se $Y \sim N(\mu, \sigma^2)$, então $X = e^Y$ é LogNormal. Sempre positiva, assimétrica à direita, surge naturalmente de processos multiplicativos (salário = base $\times$ promoções $\times$ ajustes). Mediana $= e^\mu$.

**Gamma** $\text{Gamma}(\alpha, \beta)$: Sempre positiva, forma flexível controlada por $\alpha$. Para $\alpha < 1$, altamente assimétrica; para $\alpha \gg 1$, quase simétrica. Natural para custos de "acumulação" (horas extras ao longo do mês).

**Pareto** $\text{Pareto}(\alpha, x_m)$: A distribuição de lei de potência. Cauda pesada: $P(X > x) = (x_m/x)^\alpha$ decai polinomialmente, não exponencialmente. Para modelar custos extremos (rescisões milionárias). Para $\alpha \leq 2$, variância infinita.

**Weibull** $\text{Weibull}(k, \lambda)$: Forma flexível, CDF em forma fechada. A função de risco pode ser crescente ($k > 1$), constante ($k = 1$, = Exponencial) ou decrescente ($k < 1$). Útil para custos de "tempo até evento".

### Tabela Comparativa

| Propriedade | Normal | LogNormal | Gamma | Pareto | Weibull |
|-------------|--------|-----------|-------|--------|---------|
| Suporte | $(-\infty, \infty)$ | $(0, \infty)$ | $(0, \infty)$ | $[x_m, \infty)$ | $[0, \infty)$ |
| Assimetria | 0 | $> 0$ sempre | $2/\sqrt{\alpha}$ | pesada | depende de $k$ |
| Cauda | Leve | Sub-exponencial | Leve | **Pesada** | Leve |
| MGF existe? | Sim | Não | Sim | Não | Série |

**Frame mental:** *A Normal assume simetria; a realidade dos custos é assimétrica.* A distribuição Normal **não** é candidata primária para nenhum componente individual de custo. Pode ser apropriada para o *total* orçamentário (pelo TLC), mas não para formas individuais.

![Distribuições candidatas ajustadas aos mesmos dados salariais](../figures/distribution_zoo.png)

---

## 5. Estimação por Máxima Verossimilhança

Temos famílias candidatas. Agora precisamos de uma forma principiada de escolher os *parâmetros* de cada família a partir dos dados observados. A Estimação por Máxima Verossimilhança (MLE) é o instrumento central: oferece estimativas pontuais ótimas, erros-padrão automáticos via informação de Fisher e a fundação para o framework de comparação de modelos da Seção 6.

### O Problema de Estimação

Dado um conjunto de dados $x_1, \ldots, x_n$ e uma família de distribuições $f(x \mid \theta)$, queremos encontrar o parâmetro $\theta$ que melhor explica os dados observados.

### A Função de Verossimilhança

$$L(\theta \mid \mathbf{x}) = \prod_{i=1}^n f(x_i \mid \theta)$$

A log-verossimilhança (numericamente estável):

$$\ell(\theta) = \sum_{i=1}^n \log f(x_i \mid \theta)$$

O **estimador de máxima verossimilhança** (MLE) maximiza $\ell(\theta)$:

$$\hat{\theta}_{MLE} = \arg\max_\theta \ell(\theta)$$

> **Intuição.** A verossimilhança pergunta: *"quão plausível é este parâmetro, dado o que observei?"* O MLE escolhe o parâmetro que torna os dados observados mais plausíveis. Para uma Normal, isso resulta na média e variância amostrais — confirmando que a estatística do dia a dia é, implicitamente, MLE sob suposição Normal.

### A Função Score e Informação de Fisher

A **função score** é o gradiente da log-verossimilhança:

$$S(\theta) = \frac{\partial \ell}{\partial \theta}$$

**Propriedade fundamental**: $E[S(\theta_0)] = 0$ no parâmetro verdadeiro. Demonstração: diferencie $\int f(x|\theta) dx = 1$ sob o sinal de integral.

A **informação de Fisher** mede a curvatura da log-verossimilhança:

$$I(\theta) = \text{Var}[S(\theta)] = -E\left[\frac{\partial^2 \ell}{\partial \theta^2}\right]$$

> **Intuição.** *Curvatura é precisão.* Um pico estreito e agudo da log-verossimilhança significa que os dados identificam fortemente o parâmetro; um pico achatado significa que muitos parâmetros são quase igualmente plausíveis. A informação de Fisher formaliza essa ideia.

### Normalidade Assintótica

Para $n$ grande, o MLE é aproximadamente Normal:

$$\hat{\theta}_{MLE} \dot{\sim} N\left(\theta_0, \frac{1}{n \cdot I_1(\theta_0)}\right)$$

Isso nos dá intervalos de confiança automáticos:

$$\hat{\theta} \pm z_{\alpha/2} \cdot \frac{1}{\sqrt{n \cdot I_1(\hat{\theta})}}$$

Na prática: você reporta não apenas a estimativa pontual mas também sua incerteza — e essa incerteza encolhe como $1/\sqrt{n}$ à medida que mais dados chegam.

### MLE para Nossas Distribuições

- **Normal**: $\hat{\mu} = \bar{x}$, $\hat{\sigma}^2 = \frac{1}{n}\sum(x_i - \bar{x})^2$ (forma fechada)
- **LogNormal**: $\hat{\mu} = \overline{\log x}$, $\hat{\sigma}^2 = \text{Var}(\log x)$ (forma fechada, via transformação logarítmica)
- **Gamma**: $\hat{\beta} = \hat{\alpha}/\bar{x}$, $\hat{\alpha}$ via solução numérica (equação envolvendo digamma)
- **Pareto** ($x_m$ conhecido): $\hat{\alpha} = n / \sum \log(x_i/x_m)$ (forma fechada)
- **Weibull**: ambos os parâmetros via otimização numérica

![Convergência do MLE: estimativas convergem para os parâmetros verdadeiros à medida que n cresce](../figures/mle_convergence.png)

---

## 6. Comparação de Modelos

Agora podemos ajustar qualquer candidata aos dados. Mas ajustar sozinho não nos diz qual família é a correta — e um modelo com mais parâmetros sempre se ajusta melhor aos dados de treinamento. A pergunta vira: como selecionar entre modelos ajustados sem premiar complexidade pela complexidade?

### O Problema de Seleção

Ajustamos várias distribuições aos mesmos dados. Cada uma produz uma log-verossimilhança maximizada $\ell(\hat{\theta})$. Qual modelo escolher?

Não podemos simplesmente escolher o maior $\ell(\hat{\theta})$ — modelos mais complexos sempre se ajustam melhor aos dados de treinamento (overfitting).

### Divergência KL: Medindo Distância entre Distribuições

A divergência de Kullback-Leibler mede a "perda de informação" ao usar $q$ para aproximar $p$:

$$D_{KL}(p \| q) = \int p(x) \log\frac{p(x)}{q(x)} dx \geq 0$$

com igualdade se e somente se $p = q$ quase certamente (demonstrado via desigualdade de Jensen).

> **Intuição.** *A divergência KL é a penalidade esperada de surpresa por usar o modelo errado.* Se $q$ está próximo de $p$, predições são apenas levemente piores; se $q$ está longe de $p$, a surpresa se acumula a cada observação. O AIC é, essencialmente, um estimador dessa penalidade.

### AIC: Critério de Informação de Akaike

O AIC estima a divergência KL esperada, corrigindo o viés otimista da log-verossimilhança de treinamento:

$$\text{AIC} = -2\ell(\hat{\theta}) + 2k$$

onde $k$ é o número de parâmetros. Menor AIC = melhor modelo.

**Pesos de Akaike**: $w_i = e^{-\Delta_i/2} / \sum_j e^{-\Delta_j/2}$ — a probabilidade aproximada de que o modelo $i$ é o melhor entre os candidatos.

### BIC: Critério de Informação Bayesiano

Derivado da aproximação de Laplace à evidência marginal Bayesiana:

$$\text{BIC} = -2\ell(\hat{\theta}) + k\log n$$

Penaliza complexidade mais fortemente que o AIC para $n > 7$. O BIC é consistente: seleciona o modelo verdadeiro quando $n \to \infty$.

**Frame mental:** *AIC para predição, BIC para identificação.* Quando discordam, a escolha certa depende da pergunta sendo respondida.

### Testes de Aderência

- **Kolmogorov-Smirnov (KS)**: compara a CDF empírica com a teórica. Sensível a diferenças no centro.
- **Anderson-Darling (AD)**: como KS, mas com mais peso nas caudas. Crucial para modelagem de custos onde as caudas importam.

### Teste da Razão de Verossimilhanças

Para modelos aninhados ($M_0 \subset M_1$):

$$\Lambda = -2[\ell(\hat{\theta}_0) - \ell(\hat{\theta}_1)] \xrightarrow{d} \chi^2(\Delta k)$$

Rejeitamos o modelo restrito se $\Lambda$ exceder o valor crítico.

![Comparação de modelos: AIC, BIC e pesos de Akaike](../figures/model_comparison.png)

---

## 7. Modelos de Mistura e Multimodalidade

Até aqui cada componente foi tratado como uma única distribuição. Mas dados salariais reais violam essa suposição imediatamente: um único ajuste a uma equipe júnior/sênior mistura duas populações e produz um modelo que não representa nenhuma. Modelos de mistura estendem o framework para lidar com isso diretamente.

### O Problema

Dados salariais frequentemente são bimodais: juniores (~ R\$ 8.000) e seniores (~ R\$ 18.000) formam clusters distintos. Nenhuma distribuição unimodal captura essa estrutura.

Se ajustarmos uma única Normal, obtemos média ~ R\$ 12.200 com variância inflada. **O "colaborador médio" a R\$ 12.200 não existe em nenhum dos clusters — a média é um artefato matemático.**

### Modelo de Mistura Gaussiana (GMM)

$$f(x) = \sum_{k=1}^K \pi_k \cdot \mathcal{N}(x \mid \mu_k, \sigma_k^2)$$

onde $\pi_k \geq 0$, $\sum_k \pi_k = 1$ são os pesos de mistura.

### O Algoritmo EM

O MLE direto falha porque o $\log$ de uma soma não se decompõe. O algoritmo Expectation-Maximization resolve isso iterativamente:

**E-step**: Compute as responsabilidades (probabilidade de cada observação pertencer a cada componente):

$$\gamma_{ik} = \frac{\pi_k \cdot \mathcal{N}(x_i \mid \mu_k, \sigma_k^2)}{\sum_j \pi_j \cdot \mathcal{N}(x_i \mid \mu_j, \sigma_j^2)}$$

**M-step**: Atualize os parâmetros usando as responsabilidades como pesos:

$$\pi_k^{new} = \frac{N_k}{n}, \quad \mu_k^{new} = \frac{\sum_i \gamma_{ik} x_i}{N_k}, \quad \sigma_k^{2,new} = \frac{\sum_i \gamma_{ik}(x_i - \mu_k)^2}{N_k}$$

onde $N_k = \sum_i \gamma_{ik}$.

### Convergência Monótona

O EM garante $\ell(\theta^{(t+1)}) \geq \ell(\theta^{(t)})$ (demonstrado via desigualdade de Jensen e o ELBO). Converge para um máximo local — usamos múltiplas inicializações aleatórias para mitigar.

### Escolha de K

Usamos BIC para selecionar o número de componentes: ajuste GMMs com $K = 1, 2, 3, \ldots$ e escolha o $K$ que minimiza BIC.

### O Custo Orçamentário de Ignorar Bimodalidade

No Experimento D, ignorar bimodalidade e forçar uma Normal única em uma mistura 60% júnior / 40% sênior:
- Infla o desvio-padrão estimado em ~40%
- Distorce o VaR(95%) em R\$ 1.500–2.500 por colaborador
- Para uma equipe de 50 pessoas, são R\$ 75 mil–125 mil de reserva mal alocada

**Frame mental:** *Em distribuições bimodais, a média não representa ninguém.*

![Detecção de mistura: GMM identifica estrutura bimodal](../figures/mixture_detection.png)

---

## 8. Caudas Pesadas e Custos Extremos

Modelos de mistura tratam o *centro* da distribuição. Mas os erros de modelagem mais consequentes vivem nas *caudas* — os eventos raros-mas-catastróficos onde os orçamentos efetivamente quebram. Esta seção é o coração do artigo.

### O Clímax Prático

**Se você usa uma distribuição Normal para custos de rescisão, está sistematicamente sub-reservando. Isso não é opinião. É um fato matemático.**

### Leve vs Pesada: Definição Formal

Uma distribuição é de **cauda leve** se sua função geradora de momentos existe para algum $t > 0$: $M_X(t) = E[e^{tX}] < \infty$. Equivalentemente, $P(X > x)$ decai pelo menos exponencialmente.

Uma distribuição é de **cauda pesada** se $M_X(t) = \infty$ para todo $t > 0$. A função de sobrevivência decai mais lentamente que qualquer exponencial.

### O Comportamento da Cauda Pareto

Para $X \sim \text{Pareto}(\alpha, x_m)$:

$$P(X > x) = \left(\frac{x_m}{x}\right)^\alpha$$

Isso decai **polinomialmente**. Compare com a Normal, que decai como $e^{-x^2/2}$ (super-exponencialmente).

**Razão de afinamento da cauda:**
- Pareto: $P(X > 2c) / P(X > c) = 2^{-\alpha}$ (constante!)
- Normal: a mesma razão decai exponencialmente

> **Frame mental.** *Risco de cauda é onde orçamentos falham, não onde médias vivem.* Em um mundo Normal, dobrar o limite torna o evento astronomicamente mais raro. Em um mundo Pareto, dobrar o limite reduz a probabilidade por um fator fixo — independente de onde você começou.

### Estimador de Hill

Para estimar o índice de cauda $\alpha$ a partir de dados:

$$\hat{\alpha}_{Hill} = \left[\frac{1}{k}\sum_{i=1}^k \log\frac{X_{(n-i+1)}}{X_{(n-k)}}\right]^{-1}$$

Baseado nos $k$ maiores valores observados. Plote $\hat{\alpha}$ vs $k$ e procure uma região estável.

### Impacto Orçamentário: A Subestimação Catastrófica

Custos de rescisão: Pareto($\alpha = 2.5$, $x_m = 10.000$) vs Normal com mesmos momentos:

| Métrica | Pareto | Normal | Razão |
|---------|--------|--------|-------|
| $P(X > 50.000)$ | 1,79% | 0,013% | **138x** |
| $P(X > 100.000)$ | 0,032% | $\approx 0$ | **>1000x** |
| ES(99%) | R\$ 105.160 | R\$ 55.297 | **1,9x** |

**Tradução executiva:** o modelo Normal diz que um evento de rescisão de R\$ 50.000 acontece uma vez a cada 77.000 casos. A Pareto diz que acontece uma vez a cada 56 casos. *É por isso que uma única rodada de demissões pode estourar o orçamento de um trimestre.*

### Value at Risk (VaR) e Expected Shortfall (ES)

**VaR** ao nível $p$: o quantil $p$-ésimo. "Temos $p$% de confiança que o custo não excederá VaR."

**ES** (CVaR): a perda média dado que excedeu o VaR. "Se os custos excederem o VaR, quão ruim é em média?"

Para a Pareto: $\text{ES}_p = \frac{\alpha}{\alpha - 1} \cdot \text{VaR}_p$ — sempre um múltiplo fixo do VaR.

![Comparação de risco de cauda: Normal vs Pareto](../figures/tail_risk_comparison.png)

---

## 9. Experimentos e Resultados

A teoria acima prevê consequências específicas. Esta seção testa essas previsões através de experimentos controlados. Cada experimento isola uma afirmação, roda em dados sintéticos com verdade conhecida e mede o gap entre modelagem correta e incorreta.

### Por Que Dados Sintéticos?

Todos os experimentos deste artigo usam dados sintéticos gerados a partir de distribuições conhecidas. Esta é uma escolha deliberada, não uma limitação:

- **Controle do ground truth:** sabemos exatamente qual distribuição gerou os dados, então conseguimos medir quão bem cada candidata recupera essa verdade. Com dados reais, o modelo "verdadeiro" é desconhecido.
- **Reprodutibilidade:** todo experimento usa seeds aleatórios fixos. Qualquer pessoa pode rodar o código e obter figuras e números idênticos.
- **Isolamento de efeitos:** mudando um parâmetro por vez (tamanho amostral, distribuição verdadeira, composição da mistura), atribuímos efeitos a causas específicas — não a qualidade dos dados, ruído de NDA ou artefatos amostrais.

Isto troca validade externa (vai funcionar nos dados do *seu* RH?) por validade interna (o framework faz o que afirma?). O repositório associado documenta como aplicar o mesmo pipeline a dados reais.

### Experimento A: O Zoológico de Distribuições

- **Objetivo:** mostrar como cinco famílias candidatas se adaptam aos mesmos dados assimétricos à direita.
- **Setup:** $n = 2.000$ amostras de LogNormal($\mu = 9.1$, $\sigma = 0.4$); ajustar Normal, LogNormal, Gamma e Weibull via MLE.
- **Métrica:** sobreposição visual das PDFs ajustadas contra o histograma empírico.
- **Resultado:** LogNormal captura assimetria perfeitamente; Normal erra tanto no pico quanto na cauda; Gamma e Weibull aproximam mas subestimam a cauda direita.

### Experimento B: Convergência do MLE

- **Objetivo:** verificar empiricamente consistência e normalidade assintótica do MLE.
- **Setup:** 200 replicações em cada $n \in \{20, 50, 100, \ldots, 10.000\}$, ajustando LogNormal($9.1, 0.4$).
- **Métrica:** média e desvio-padrão de $\hat{\mu}$ entre replicações; comparação com SE teórico = $\sigma / \sqrt{n}$.
- **Resultado:** estimativas convergem para o parâmetro verdadeiro; SD empírico bate com o decaimento teórico $1/\sqrt{n}$ ao longo de três ordens de grandeza.

### Experimento C: O Custo da Distribuição Errada

- **Objetivo:** quantificar o erro orçamentário ao ajustar Normal a dados LogNormal.
- **Setup:** gerar $n = 1.000$ de LogNormal; ajustar Normal e LogNormal; simular 200 mil amostras de cada modelo ajustado.
- **Métrica:** $P(\text{custo} > \text{teto})$ em múltiplos da média; gap de VaR(99%) e ES(99%).
- **Resultado:** Normal subestima $P(X > 2 \cdot \text{média})$ em ~3x e $P(X > 3 \cdot \text{média})$ em ~8x. Para uma equipe de 50 pessoas, isto equivale a R\$ 100 mil–150 mil de reserva insuficiente.

![Impacto da distribuição errada no orçamento](../figures/wrong_distribution_impact.png)

### Experimento D: Detecção de Mistura

- **Objetivo:** mostrar que GMM com seleção via BIC recupera estrutura bimodal oculta.
- **Setup:** 60% amostrado de $N(8000, 1500^2)$, 40% de $N(18000, 2500^2)$; ajustar GMMs com $K \in \{1, 2, 3, 4\}$.
- **Métrica:** BIC ao longo de $K$; pesos, médias e desvios-padrão recuperados.
- **Resultado:** BIC favorece fortemente $K = 2$. Parâmetros recuperados ficam dentro de 5% dos valores verdadeiros. Ajuste de Normal única produz uma média que não representa colaborador real.

### Experimento E: Risco de Cauda Pesada

- **Objetivo:** medir o gap de probabilidade de cauda entre Pareto e Normal com mesmos dois primeiros momentos.
- **Setup:** Pareto($\alpha = 2.5$, $x_m = 10.000$) vs Normal com momentos casados em $\mu = 16.667$, $\sigma = 14.907$.
- **Métrica:** $P(X > x)$ em limites de R\$ 30 mil a R\$ 200 mil; VaR e ES analíticos a 90%, 95%, 99%.
- **Resultado:** Normal subestima $P(X > 50K)$ por 138x e ES(99%) por ~1,9x. O "evento raro" sob Normal é rotineiro sob Pareto.

### Experimento F: Comparação de Modelos

- **Objetivo:** validar que AIC, BIC e KS conjuntamente identificam a distribuição verdadeira.
- **Setup:** $n = 500$ de LogNormal; ajustar todas as cinco candidatas; computar critérios e testes de aderência.
- **Métrica:** pesos de Akaike, ranking de BIC, p-valores do KS.
- **Resultado:** LogNormal vence com peso de Akaike > 96%. Normal é decisivamente rejeitada. KS confirma que LogNormal é a única candidata que passa no teste de aderência.

### Experimento G: Pipeline End-to-End

- **Objetivo:** demonstrar o fluxo completo em uma equipe sintética de 50 pessoas.
- **Setup:** gerar dados de salário, hora extra, rescisão e contratação; ajustar todas as candidatas a cada componente; rankear via AIC/BIC; computar impacto orçamentário.
- **Métrica:** seleção automática do melhor modelo por componente; VaR e reserva resultantes.
- **Resultado:** o pipeline identifica corretamente LogNormal para salário e Pareto para rescisão. Reserva total no VaR 99% difere em R\$ 100 mil–150 mil da estimativa baseline-Normal.

![Pipeline end-to-end: dados → ajuste → seleção → impacto orçamentário](../figures/full_pipeline.png)

---

## 10. Framework Prático

A teoria e os experimentos acima apontam para um procedimento de decisão concreto. Esta seção condensa tudo em um fluxo de cinco passos que qualquer analista pode aplicar aos seus próprios dados.

### Árvore de Decisão para Seleção de Distribuição

**Passo 1: Visualize**
- Histograma + Q-Q plot
- Os dados são assimétricos? Multimodais? Há valores extremos?

**Passo 2: Ajuste candidatas**
- Use MLE para ajustar 3-5 famílias de distribuição
- Verifique convergência e razoabilidade dos parâmetros

**Passo 3: Compare**
- Compute AIC e BIC para todas as candidatas
- Compute pesos de Akaike — há um vencedor claro?
- Use AIC para predição, BIC para identificar o modelo "verdadeiro"

**Passo 4: Valide**
- Teste de aderência (Anderson-Darling para caudas)
- Q-Q plot do modelo vencedor
- O modelo vencedor passa nos testes?

**Passo 5: Quantifique o impacto**
- Compute VaR e ES sob o modelo selecionado
- Compare com o que a Normal teria previsto
- Traduza a diferença em R\$ de reserva orçamentária

### Armadilhas Comuns

1. **Usar Normal por padrão** → subestima caudas sistematicamente
2. **Ignorar multimodalidade** → variância inflada, média sem sentido
3. **Olhar só a média** → ignora toda a informação da forma
4. **Amostra pequena** → use AICc, não AIC
5. **Não validar** → o modelo com melhor AIC pode ainda assim não se ajustar bem

### Implementação Mínima Viável

Se você tem uma tarde e um conjunto de dados, este é o caminho mais curto para gerar valor:

1. **Ajuste dois modelos:** LogNormal (para dados positivos assimétricos) e Pareto (para dados de cauda pesada).
2. **Compare via AIC:** o menor vence. Se próximos, escolha LogNormal pela simplicidade.
3. **Compute VaR(95%) e VaR(99%)** sob o vencedor.
4. **Compute o mesmo sob um ajuste Normal.**
5. **Reporte o delta.** Esse número é a sub-reserva caso a Normal tivesse sido o default.

Esses cinco passos cobrem a maioria das decisões reais de modelagem de custo. O framework completo só adiciona rigor onde os dados exigem.

---

## 11. Limitações

O framework acima é deliberadamente delimitado. As limitações abaixo não são falhas do método — são fronteiras dentro das quais ele opera.

**Sensibilidade ao tamanho amostral.** Com $n < 50$, as estimativas MLE são ruidosas e o AIC pode escolher o modelo errado com probabilidade não-trivial. AICc ajuda mas não elimina o problema. Para equipes pequenas ou históricos curtos, recomenda-se bootstrap paramétrico em vez de intervalos de confiança assintóticos.

**Risco de modelo persiste.** Escolher a melhor entre cinco candidatas não garante que alguma delas seja correta. Testes de aderência protegem contra má-especificação grosseira mas não detectam uma sexta família não considerada. O framework reduz risco de modelo; não o elimina.

**Suposição de independência.** Cada componente de custo é modelado independentemente. Na realidade, componentes são correlacionados: uma onda de demissões simultaneamente reduz custos de contratação e infla rescisões. Modelar essas dependências exige métodos multivariados (cópulas, distribuições conjuntas) cobertos no artigo complementar de Monte Carlo.

**Distribuições estáticas.** Este artigo assume que as distribuições são estáveis no tempo. Distribuições salariais derivam com inflação, mudanças de mercado e mudanças organizacionais. Métodos de séries temporais (modelos de espaço de estados, regime-switching) estão fora do escopo.

**Complicações de dados reais.** Dados sintéticos são limpos. Dados reais de RH têm censura (colaboradores ainda ativos quando medidos), truncamento (apenas rescisões acima do mínimo legal são registradas), dados faltantes e ruído de relatório. O framework se aplica, mas pré-processamento importa.

**Contexto de decisão omitido.** Um "modelo melhor" pelo AIC nem sempre é uma decisão de negócio melhor. Apetite por risco, capital regulatório e conservadorismo de stakeholders podem empurrar para a Normal mesmo quando ela ajusta pior — porque seus outputs são mais familiares. O framework informa a decisão; não a substitui.

---

## 12. Conclusão

### A Distribuição É o Modelo

A suposição distribucional não é um detalhe técnico — é a decisão de modelagem mais importante que um analista de custos faz. Todo cálculo a jusante (média, variância, intervalos de confiança, reservas) herda essa escolha.

### Principais Conclusões

1. **Custos de pessoal são estruturalmente não-Normais**: assimétricos (LogNormal), de cauda pesada (Pareto), e frequentemente multimodais (GMM). A Normal é a exceção, não a regra.

2. **O MLE fornece o framework principiado** para ajuste: parâmetros ótimos, erros-padrão automáticos, e uma base teórica sólida para comparação via AIC/BIC.

3. **O impacto de errar é mensurável e substancial**: a Normal subestima a probabilidade de custos extremos por fatores de 100x ou mais. Para uma equipe de 50 pessoas, isso pode representar centenas de milhares de reais em reserva insuficiente.

### Próximos Passos

Este artigo estabelece o framework de seleção distribucional. O artigo complementar (Monte Carlo) mostra como usar essas distribuições para simular o custo total de uma equipe, incluindo correlações entre componentes e análise de cenários.

O código completo está disponível no repositório associado, com dados sintéticos reprodutíveis e figuras em qualidade de publicação.

---

## Referências

- Casella, G. & Berger, R. (2002). *Statistical Inference*. Duxbury.
- Burnham, K. & Anderson, D. (2002). *Model Selection and Multimodel Inference*. Springer.
- Bishop, C. (2006). *Pattern Recognition and Machine Learning*. Springer.
- Embrechts, P., Klüppelberg, C. & Mikosch, T. (1997). *Modelling Extremal Events*. Springer.
- McLachlan, G. & Peel, D. (2000). *Finite Mixture Models*. Wiley.
