# A Forma do Que Você Vai Gastar

## Modelagem Probabilística de Custos de Pessoal — da Seleção de Distribuições ao Impacto Orçamentário

---

## 1. Introdução: Por Que Distribuições Importam

Um orçamento é uma declaração probabilística disfarçada de planilha. Quando um analista projeta custos de pessoal, está implicitamente assumindo uma distribuição de probabilidade para cada componente — salários, horas extras, rescisões, contratações. Na maioria das organizações, essa suposição é a distribuição Normal: simétrica, com caudas leves, bem-comportada.

O problema é que custos de pessoal não são Normais.

Salários são assimétricos à direita: a maioria dos colaboradores recebe salários moderados, enquanto poucos executivos puxam a média para cima. Custos de rescisão são de cauda pesada: a maioria é moderada, mas alguns poucos casos envolvem valores extremos que podem comprometer o orçamento de um trimestre inteiro. Dados salariais frequentemente são multimodais: clusters de juniores, plenos e seniores formam picos distintos que nenhuma distribuição unimodal pode capturar.

**A distribuição que você assume é o seu modelo.** Todo o resto — média, variância, intervalos de confiança, estimativas de risco — flui dessa escolha. Uma distribuição errada não é uma nuance de modelagem; é um viés sistemático que se propaga por todo cálculo a jusante.

$$\text{Distribuição errada} \rightarrow \text{parâmetros errados} \rightarrow \text{orçamento errado} \rightarrow \text{decisões erradas}$$

Este artigo apresenta um framework rigoroso para seleção de distribuições em modelagem de custos. Derivamos a Estimação por Máxima Verossimilhança (MLE) a partir dos fundamentos, ajustamos cinco famílias de distribuições candidatas a dados sintéticos de custos, e usamos critérios informacionais (AIC, BIC) e testes de aderência para selecionar o melhor modelo. Quantificamos o impacto orçamentário de errar na escolha — em um experimento, a subestimação da probabilidade de exceder o dobro do custo esperado chega a um fator de 138x.

### Relação com o Artigo de Monte Carlo

Este artigo é o segundo de um par complementar. O primeiro artigo responde "como simular o custo total dado distribuições assumidas" (Monte Carlo). Este artigo responde a pergunta anterior: "quais distribuições assumir em primeiro lugar?"

---

## 2. Os Componentes de Custo

### O Modelo

Representamos cada componente de custo como uma variável aleatória com propriedades distribucionais distintas. O modelo é genérico, mínimo e expansível.

| Componente | Símbolo | Candidatas | Justificativa |
|------------|---------|------------|---------------|
| Salário base | $S_i$ | LogNormal, Gamma, Mistura(Normal) | Assimétrico à direita; multimodal se há clusters |
| Custo de hora extra | $C_{ot}$ | LogNormal, Gamma | Assimétrico, sempre positivo |
| Custo de rescisão | $C_{sev}$ | **Pareto**, LogNormal | Cauda pesada: poucos casos extremamente caros |
| Custo de contratação | $C_h$ | LogNormal, Gamma | Variável (fees de recrutador, relocação) |
| Multiplicador de benefícios | $\beta_i$ | Uniform, Beta | Limitado: tipicamente 1,3x–2,2x do salário |

### Exemplo Concreto: Equipe de 50 Pessoas

Considere uma equipe de TI com 50 colaboradores. Usando parâmetros calibrados para o mercado brasileiro:

- **Salários**: LogNormal($\mu = 9.1$, $\sigma = 0.4$), mediana ~ R\$ 9.000/mês
- **Horas extras**: Gamma($\alpha = 4$, $\beta = 1/30$), média ~ R\$ 120/hora
- **Rescisões**: Pareto($\alpha = 2.5$, $x_m = 10.000$), ~3 eventos/ano
- **Contratações**: LogNormal($\mu = 9.5$, $\sigma = 0.7$), ~5 contratações/ano

O custo anual total esperado é de aproximadamente R\$ 6,0–6,5 milhões. A questão crucial: a *variância* desse total depende criticamente de quais distribuições você assume. Suposições Normais produzem um intervalo de confiança estreito; suposições corretas de cauda pesada produzem um intervalo muito mais amplo.

---

## 3. Famílias de Distribuições

### As Cinco Candidatas

Para cada componente de custo, consideramos cinco famílias de distribuições, cada uma com propriedades de forma distintas.

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

A distribuição Normal **não** é candidata primária para nenhum componente individual de custo. Pode ser apropriada para o *total* orçamentário (pelo TLC), mas não para formas individuais.

![Distribuições candidatas ajustadas aos mesmos dados salariais](../figures/distribution_zoo.png)

---

## 4. Estimação por Máxima Verossimilhança

### O Problema de Estimação

Dado um conjunto de dados $x_1, \ldots, x_n$ e uma família de distribuições $f(x \mid \theta)$, queremos encontrar o parâmetro $\theta$ que melhor explica os dados observados.

### A Função de Verossimilhança

$$L(\theta \mid \mathbf{x}) = \prod_{i=1}^n f(x_i \mid \theta)$$

A log-verossimilhança (numericamente estável):

$$\ell(\theta) = \sum_{i=1}^n \log f(x_i \mid \theta)$$

O **estimador de máxima verossimilhança** (MLE) maximiza $\ell(\theta)$:

$$\hat{\theta}_{MLE} = \arg\max_\theta \ell(\theta)$$

### A Função Score e Informação de Fisher

A **função score** é o gradiente da log-verossimilhança:

$$S(\theta) = \frac{\partial \ell}{\partial \theta}$$

**Propriedade fundamental**: $E[S(\theta_0)] = 0$ no parâmetro verdadeiro. Demonstração: diferencie $\int f(x|\theta) dx = 1$ sob o sinal de integral.

A **informação de Fisher** mede a curvatura da log-verossimilhança:

$$I(\theta) = \text{Var}[S(\theta)] = -E\left[\frac{\partial^2 \ell}{\partial \theta^2}\right]$$

Mais informação = mais curvatura = estimativas mais precisas.

### Normalidade Assintótica

Para $n$ grande, o MLE é aproximadamente Normal:

$$\hat{\theta}_{MLE} \dot{\sim} N\left(\theta_0, \frac{1}{n \cdot I_1(\theta_0)}\right)$$

Isso nos dá intervalos de confiança automáticos:

$$\hat{\theta} \pm z_{\alpha/2} \cdot \frac{1}{\sqrt{n \cdot I_1(\hat{\theta})}}$$

### MLE para Nossas Distribuições

- **Normal**: $\hat{\mu} = \bar{x}$, $\hat{\sigma}^2 = \frac{1}{n}\sum(x_i - \bar{x})^2$ (forma fechada)
- **LogNormal**: $\hat{\mu} = \overline{\log x}$, $\hat{\sigma}^2 = \text{Var}(\log x)$ (forma fechada)
- **Gamma**: $\hat{\beta} = \hat{\alpha}/\bar{x}$, $\hat{\alpha}$ via solução numérica (equação envolvendo digamma)
- **Pareto** ($x_m$ conhecido): $\hat{\alpha} = n / \sum \log(x_i/x_m)$ (forma fechada)
- **Weibull**: ambos os parâmetros via otimização numérica

![Convergência do MLE: estimativas convergem para os parâmetros verdadeiros à medida que n cresce](../figures/mle_convergence.png)

---

## 5. Comparação de Modelos

### O Problema de Seleção

Ajustamos várias distribuições aos mesmos dados. Cada uma produz uma log-verossimilhança maximizada $\ell(\hat{\theta})$. Qual modelo escolher?

Não podemos simplesmente escolher o maior $\ell(\hat{\theta})$ — modelos mais complexos sempre se ajustam melhor aos dados de treinamento (overfitting).

### Divergência KL: Medindo Distância entre Distribuições

A divergência de Kullback-Leibler mede a "perda de informação" ao usar $q$ para aproximar $p$:

$$D_{KL}(p \| q) = \int p(x) \log\frac{p(x)}{q(x)} dx \geq 0$$

com igualdade se e somente se $p = q$ quase certamente (demonstrado via desigualdade de Jensen).

### AIC: Critério de Informação de Akaike

O AIC estima a divergência KL esperada, corrigindo o viés otimista da log-verossimilhança de treinamento:

$$\text{AIC} = -2\ell(\hat{\theta}) + 2k$$

onde $k$ é o número de parâmetros. Menor AIC = melhor modelo.

**Pesos de Akaike**: $w_i = e^{-\Delta_i/2} / \sum_j e^{-\Delta_j/2}$ — a probabilidade aproximada de que o modelo $i$ é o melhor entre os candidatos.

### BIC: Critério de Informação Bayesiano

Derivado da aproximação de Laplace à evidência marginal Bayesiana:

$$\text{BIC} = -2\ell(\hat{\theta}) + k\log n$$

Penaliza complexidade mais fortemente que o AIC para $n > 7$. O BIC é consistente: seleciona o modelo verdadeiro quando $n \to \infty$.

### Testes de Aderência

- **Kolmogorov-Smirnov (KS)**: compara a CDF empírica com a teórica. Sensível a diferenças no centro.
- **Anderson-Darling (AD)**: como KS, mas com mais peso nas caudas. Crucial para modelagem de custos onde as caudas importam.

### Teste da Razão de Verossimilhanças

Para modelos aninhados ($M_0 \subset M_1$):

$$\Lambda = -2[\ell(\hat{\theta}_0) - \ell(\hat{\theta}_1)] \xrightarrow{d} \chi^2(\Delta k)$$

Rejeitamos o modelo restrito se $\Lambda$ exceder o valor crítico.

![Comparação de modelos: AIC, BIC e pesos de Akaike](../figures/model_comparison.png)

---

## 6. Modelos de Mistura e Multimodalidade

### O Problema

Dados salariais frequentemente são bimodais: juniores (~ R\$ 8.000) e seniores (~ R\$ 18.000) formam clusters distintos. Nenhuma distribuição unimodal captura essa estrutura.

Se ajustarmos uma única Normal, obtemos média ~ R\$ 12.200 com variância inflada. O "colaborador médio" a R\$ 12.200 não existe em nenhum dos clusters — a média é enganosa.

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

![Detecção de mistura: GMM identifica estrutura bimodal](../figures/mixture_detection.png)

---

## 7. Caudas Pesadas e Custos Extremos

### O Clímax Prático

Este é o ponto central do artigo: **se você usa uma distribuição Normal para custos de rescisão, está sistematicamente sub-reservando.**

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

A Normal diz que um evento de R\$ 50.000 acontece uma vez a cada 77.000 casos. A Pareto diz que acontece uma vez a cada 56 casos. As implicações para reservas orçamentárias são radicalmente diferentes.

### Value at Risk (VaR) e Expected Shortfall (ES)

**VaR** ao nível $p$: o quantil $p$-ésimo. "Temos $p$% de confiança que o custo não excederá VaR."

**ES** (CVaR): a perda média dado que excedeu o VaR. "Se os custos excederem o VaR, quão ruim é em média?"

Para a Pareto: $\text{ES}_p = \frac{\alpha}{\alpha - 1} \cdot \text{VaR}_p$ — sempre um múltiplo fixo do VaR.

![Comparação de risco de cauda: Normal vs Pareto](../figures/tail_risk_comparison.png)

---

## 8. Experimentos e Resultados

### Experimento A: O Zoológico de Distribuições

Ajustamos todas as cinco famílias candidatas aos mesmos dados salariais sintéticos (LogNormal como verdade). A figura mostra como cada família se adapta: a LogNormal captura perfeitamente a assimetria, enquanto a Normal falha nas caudas.

### Experimento B: Convergência do MLE

Demonstramos que as estimativas MLE convergem para os parâmetros verdadeiros conforme $n$ cresce, e que os erros padrão diminuem proporcionalmente a $1/\sqrt{n}$. Com $n = 50$, as estimativas já são razoáveis; com $n = 5.000$, são essencialmente exatas.

### Experimento C: O Custo da Distribuição Errada

Ajustamos Normal a dados que são na verdade LogNormal. Resultados:
- A probabilidade de exceder 2x a média é subestimada em ~3x
- A probabilidade de exceder 3x a média é subestimada em ~8x
- O VaR a 99% é subestimado em R\$ 2.000–3.000 por colaborador
- Para uma equipe de 50 pessoas, isso representa R\$ 100.000–150.000 de reserva insuficiente

![Impacto da distribuição errada no orçamento](../figures/wrong_distribution_impact.png)

### Experimento D: Detecção de Mistura

Geramos dados bimodais (60% juniores a R\$ 8.000, 40% seniores a R\$ 18.000). O GMM com seleção por BIC identifica corretamente $K = 2$ componentes e recupera os parâmetros com boa precisão. Uma Normal única produz média de R\$ 12.000 que não representa nenhum colaborador real.

### Experimento E: Risco de Cauda Pesada

O experimento central: comparamos Pareto vs Normal para custos de rescisão. A Normal subestima $P(X > 50.000)$ por um fator de 138x. Para planejamento orçamentário, isso é a diferença entre "evento impossível" e "acontece ~2% das vezes".

### Experimento F: Comparação de Modelos

Aplicamos AIC, BIC e teste KS a dados salariais. A LogNormal vence com peso de Akaike > 96%. A Normal é decisivamente rejeitada. O Gamma compete mas fica em segundo lugar.

### Experimento G: Pipeline Completo

Demonstração end-to-end: dados de equipe → ajuste de todas as candidatas → seleção via AIC/BIC → quantificação de impacto orçamentário. O pipeline identifica automaticamente LogNormal para salários e Pareto para rescisões.

![Pipeline end-to-end: dados → ajuste → seleção → impacto orçamentário](../figures/full_pipeline.png)

---

## 9. Framework Prático

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

---

## 10. Conclusão

### A Distribuição É o Modelo

A suposição distribucional não é um detalhe técnico — é a decisão de modelagem mais importante que um analista de custos faz. Todo cálculo a jusante (média, variância, intervalos de confiança, reservas) herda essa escolha.

### Principais Conclusões

1. **Custos de pessoal são estruturalmente não-Normais**: assimétricos (LogNormal), de cauda pesada (Pareto), e frequentemente multimodais (GMM). A Normal é a exceção, não a regra.

2. **O MLE fornece o framework principiado** para ajuste: parâmetros ótimos, erros padrão automáticos, e uma base teórica sólida para comparação via AIC/BIC.

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
