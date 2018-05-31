//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//
#pragma once

#include "Basics.h"
#include "ComputationNode.h"
#include "gammacalculation.h"
#include "InputAndParamNodes.h"
#include "Sequences.h"
#include <map>
#include <string>
#include <vector>
#include <stdexcept>
#include <list>
#include <memory>
#include <queue>


namespace Microsoft { namespace MSR { namespace CNTK {

// -----------------------------------------------------------------------
// ClassificationErrorNode (label, prediction)   or ClassificationErrorNode (prediction, label)
// Performs classification and error counting.
// Result is an error rate, lower = better.
// -----------------------------------------------------------------------
    static map<size_t, size_t>TIMIT_61to39_map =
    {
        { (size_t)0, (size_t)0 },
        { (size_t)1, (size_t)12 },
        { (size_t)2, (size_t)1 },
        { (size_t)3, (size_t)0 },
        { (size_t)4, (size_t)13 },
        { (size_t)5, (size_t)1 },
        { (size_t)6, (size_t)1 },
        { (size_t)7, (size_t)2 },
        { (size_t)8, (size_t)14 },
        { (size_t)9, (size_t)15 },
        { (size_t)10, (size_t)11 },
        { (size_t)11, (size_t)16 },
        { (size_t)12, (size_t)17 },
        { (size_t)13, (size_t)11 },
        { (size_t)14, (size_t)18 },
        { (size_t)15, (size_t)19 },
        { (size_t)16, (size_t)20 },
        { (size_t)17, (size_t)5 },
        { (size_t)18, (size_t)6 },
        { (size_t)19, (size_t)7 },
        { (size_t)20, (size_t)8 },
        { (size_t)21, (size_t)11 },
        { (size_t)22, (size_t)2 },
        { (size_t)23, (size_t)21 },
        { (size_t)24, (size_t)22 },
        { (size_t)25, (size_t)23 },
        { (size_t)26, (size_t)11 },
        { (size_t)27, (size_t)11 },
        { (size_t)28, (size_t)3 },
        { (size_t)29, (size_t)3 },
        { (size_t)30, (size_t)4 },
        { (size_t)31, (size_t)4 },
        { (size_t)32, (size_t)24 },
        { (size_t)33, (size_t)25 },
        { (size_t)34, (size_t)26 },
        { (size_t)35, (size_t)11 },
        { (size_t)36, (size_t)5 },
        { (size_t)37, (size_t)6 },
        { (size_t)38, (size_t)7 },
        { (size_t)39, (size_t)8 },
        { (size_t)40, (size_t)7 },
        { (size_t)41, (size_t)27 },
        { (size_t)42, (size_t)28 },
        { (size_t)43, (size_t)29 },
        { (size_t)44, (size_t)11 },
        { (size_t)45, (size_t)11 },
        { (size_t)47, (size_t)30 },
        { (size_t)48, (size_t)31 },
        { (size_t)49, (size_t)9 },
        { (size_t)50, (size_t)32 },
        { (size_t)51, (size_t)11 },
        { (size_t)52, (size_t)33 },
        { (size_t)53, (size_t)34 },
        { (size_t)54, (size_t)10 },
        { (size_t)55, (size_t)10 },
        { (size_t)56, (size_t)35 },
        { (size_t)57, (size_t)36 },
        { (size_t)58, (size_t)37 },
        { (size_t)59, (size_t)38 },
        { (size_t)60, (size_t)9 },
        { (size_t)46, (size_t)39 },
        { (size_t)61, (size_t)40 }
    };
template <class ElemType>
class ClassificationErrorNode : public ComputationNodeNonLooping /*ComputationNode*/<ElemType>
{
    typedef ComputationNodeNonLooping<ElemType> Base; UsingComputationNodeMembersBoilerplate;
    static const std::wstring TypeName() { return L"ClassificationError"; }

public:
    DeclareConstructorFromConfig(ClassificationErrorNode);
    ClassificationErrorNode(DEVICEID_TYPE deviceId, const wstring& name)
        : Base(deviceId, name)
    {
    }

    virtual void BackpropToNonLooping(size_t /*inputIndex*/) override
    {
        LogicError("%ls operation is used for evaluation only.", OperationName().c_str());
    }

    virtual bool OutputUsedInComputingInputNodesGradients() const override
    {
        return false;
    }
    virtual bool InputUsedInComputingInputNodesGradients(size_t /*childIndex*/) const override
    {
        return false;
    }

    virtual void /*ComputationNodeNonLooping::*/ ForwardPropNonLooping() override
    {
        FrameRange fr(InputRef(0).GetMBLayout());
        InputRef(0).ValueFor(fr).VectorMax(*m_maxIndexes0, *m_maxValues, true);
        InputRef(1).ValueFor(fr).VectorMax(*m_maxIndexes1, *m_maxValues, true, m_topK);
        MaskMissingColumnsToZero(*m_maxIndexes0, InputRef(0).GetMBLayout(), fr);
        MaskMissingColumnsToZero(*m_maxIndexes1, InputRef(1).GetMBLayout(), fr);
        Value().AssignNumOfDiff(*m_maxIndexes0, *m_maxIndexes1, m_topK > 1);
#if NANCHECK
        Value().HasNan("ClassificationError");
#endif
#if DUMPOUTPUT
        Value().Print("ClassificationErrorNode");
#endif
    }

    virtual void /*ComputationNodeBase::*/ Validate(bool isFinalValidationPass) override
    {
        ValidateBinaryReduce(isFinalValidationPass);

        m_topK = 1;
        // TODO: Make topK a constructor parameter
        if (m_inputs.size() == 3)
        {
            if (Input(2)->GetSampleLayout().GetNumElements() != 1)
                InvalidArgument("%ls %ls operation requires TopK to be a scalar value.", NodeName().c_str(), OperationName().c_str());
            m_topK = static_cast<int>(Input(2)->Get00Element());
        }
    }

    virtual void UpdateFunctionMBSize() override
    {
        Base::UpdateFunctionMBSize();

        // resize the temporaries to their proper size
        size_t cols = Input(0)->Value().GetNumCols();
        m_maxIndexes0->Resize(m_topK, cols);
        m_maxIndexes1->Resize(m_topK, cols);
        m_maxValues->Resize(m_topK, cols);
    }

    virtual void CopyTo(ComputationNodeBasePtr nodeP, const std::wstring& newName, const CopyNodeFlags flags) const override
    {
        Base::CopyTo(nodeP, newName, flags);
        if (flags & CopyNodeFlags::copyNodeValue)
        {
            auto node = dynamic_pointer_cast<ClassificationErrorNode<ElemType>>(nodeP);
            node->m_maxIndexes0->SetValue(*m_maxIndexes0);
            node->m_maxIndexes1->SetValue(*m_maxIndexes1);
            node->m_maxValues->SetValue(*m_maxValues);
        }
    }
    // request matrices needed to do node function value evaluation
    virtual void RequestMatricesBeforeForwardProp(MatrixPool& matrixPool)
    {
        Base::RequestMatricesBeforeForwardProp(matrixPool);
        RequestMatrixFromPool(m_maxIndexes0, matrixPool);
        RequestMatrixFromPool(m_maxIndexes1, matrixPool);
        RequestMatrixFromPool(m_maxValues, matrixPool);
    }

    // release temp matrices that are only used by forward computation
    // don't release matrices that need to be used in the gradient computation
    virtual void ReleaseMatricesAfterForwardProp(MatrixPool& matrixPool)
    {
        Base::ReleaseMatricesAfterForwardProp(matrixPool);
        ReleaseMatrixToPool(m_maxIndexes0, matrixPool);
        ReleaseMatrixToPool(m_maxIndexes1, matrixPool);
        ReleaseMatrixToPool(m_maxValues, matrixPool);
    }

private:
    shared_ptr<Matrix<ElemType>> m_maxIndexes0, m_maxIndexes1;
    shared_ptr<Matrix<ElemType>> m_maxValues;
    int m_topK;
};

template class ClassificationErrorNode<float>;
template class ClassificationErrorNode<double>;

// -----------------------------------------------------------------------
// NDCG1EvalNode (gain, prediction, queryId)
// NDCG @ 1 
// -----------------------------------------------------------------------

template <class ElemType>
class NDCG1EvalNode : public ComputationNodeNonLooping /*ComputationNode*/<ElemType>, public NumInputs<3>
{
    typedef ComputationNodeNonLooping<ElemType> Base;
    UsingComputationNodeMembersBoilerplate;
    static const std::wstring TypeName()
    {
        return L"NDCG1Eval";
    }

public:
    DeclareConstructorFromConfig(NDCG1EvalNode);
    NDCG1EvalNode(DEVICEID_TYPE deviceId, const wstring& name)
        : Base(deviceId, name)
    {
    }

    virtual void BackpropToNonLooping(size_t /*inputIndex*/) override
    {
        LogicError("%ls operation is used for evaluation only.", OperationName().c_str());
    }

    virtual bool OutputUsedInComputingInputNodesGradients() const override
    {
        return false;
    }
    virtual bool InputUsedInComputingInputNodesGradients(size_t /*childIndex*/) const override
    {
        return false;
    }

    virtual void /*ComputationNodeNonLooping::*/ ForwardPropNonLooping() override
    {
        // Inputs:
        //      0. gain
        //      1. pred
        //      2. query id
        FrameRange fr(Input(0)->GetMBLayout());

        // Construct matrices for further computation.
        const Matrix<ElemType>& gains = Input(0)->ValueFor(fr);
        const Matrix<ElemType>& preds = Input(1)->ValueFor(fr);
        const Matrix<ElemType>& queryIds = Input(2)->ValueFor(fr);

        // Iterate through all samples
        size_t numberOfSamples = gains.GetNumCols();
        QueryUrls aqu;
        int previousQueryId = -1;
        int numberOfQueries = 0;

        // Iterate all samples and populate m_queryUrls table. 
        for (size_t i = 0; i < numberOfSamples; i++)
        {
            int queryId = (int)queryIds(0, i);
            // Samples are grouped by queries. Find all the urls 
            // belonging to each individual query.
            if (queryId != previousQueryId)
            {
                m_queryUrls.push_back(aqu);
                numberOfQueries++;
                previousQueryId = queryId;
            }

            // Get the last QueryUrls and add the Url.
            QueryUrls& qub = m_queryUrls.back();
            Url u(i, qub.m_urls.size(), preds(0, i), gains(0, i));
            qub.m_urls.push_back(u);
        }

        for (auto &qu : m_queryUrls)
        {
            std::vector<Url>& urls = qu.m_urls;
            // Urls are pre-sorted in descending order of gains.
            typename std::vector<Url>::iterator its = m_urlSorter.begin(), it = urls.begin();
            typename std::vector<Url>::iterator its0 = its;
            its = std::copy(it, urls.end(), its);
            std::sort(its0, its);

            // Set the sorted rk order to each url and 
            // the urls are still in the original order
            int rk = 0;
            for (it = its0; it != its; it++)
            {
                urls[it->m_rank0].m_rank = rk++;
            }
        }

        // calculate IRMetrics
        size_t sampleCount = 0;
        for (const auto &qu: m_queryUrls)
        {
            for (const auto &url : qu.m_urls)
            {
                (*m_urlGain0)(0, sampleCount) = url.m_gain;
                (*m_urlGain1)(0, sampleCount) = url.m_gain;
                (*m_urlDiscount0)(0, sampleCount) = (ElemType)url.m_rank0;
                (*m_urlDiscount1)(0, sampleCount) = (ElemType)url.m_rank;
                sampleCount++;
            }
        }

        // log(2+rank)
        *m_urlDiscount0 += 2.0;
        m_urlDiscount0->InplaceLog();
        *m_urlDiscount1 += 2.0;
        m_urlDiscount1->InplaceLog();
        // gain/log(2+rank)
        m_urlGain0->AssignElementDivisionOf(*m_urlGain0, *m_urlDiscount0);
        m_urlGain1->AssignElementDivisionOf(*m_urlGain1, *m_urlDiscount1);

        //Aggregate at query level.
        const Matrix<ElemType>& urlDiscountedGain0 = *m_urlGain0;
        const Matrix<ElemType>& urlDiscountedGain1 = *m_urlGain1;
        ElemType irMetricValue = 0.0;
        ElemType idealMetric = 0.0, metric = 0.0;

        // IRMetric @ 1
        for (auto &qu : m_queryUrls)
        {
            idealMetric = urlDiscountedGain0(0, qu.m_urls.begin()->m_id);
            if (idealMetric == 0.0) continue;

            for (auto &url : qu.m_urls)
            {
                if (url.m_rank == 0)
                {
                    metric = urlDiscountedGain1(0, url.m_id);
                    break;
                }
            }

            irMetricValue += (metric / idealMetric);
        }

        if (numberOfQueries == 0)
        {
            LogicError("In %ls %ls numberOfQueries==0, check your data.", NodeName().c_str(), OperationName().c_str());
        }

        irMetricValue = irMetricValue / numberOfQueries * 100 * numberOfSamples;
        Value().SetValue(irMetricValue);
    }

    virtual void /*ComputationNodeBase::*/ Validate(bool isFinalValidationPass) override
    {
        if (m_inputs.size() != 3)
            InvalidArgument("%ls operation requires three inputs instead of %d.", NodeDescription().c_str(), (int)m_inputs.size());

        if (Input(0)->NeedsGradient() == true)
            InvalidArgument("%ls %ls operation needs input type (no gradient) for the 1st input.", NodeName().c_str(), OperationName().c_str());

        if (Input(2)->NeedsGradient() == true)
            InvalidArgument("%ls %ls operation needs input type (no gradient) for the 3rd input.", NodeName().c_str(), OperationName().c_str());

        ValidateBinaryReduce(isFinalValidationPass);
    }

    virtual void UpdateFunctionMBSize() override
    {
        UpdateCounts();

        // clean up first
        if (!m_queryUrls.empty()) m_queryUrls.clear();
        if (!m_urlSorter.empty()) m_urlSorter.clear();
        if (!m_logWeights.empty()) m_logWeights.clear();

        m_urlGain0->Resize(1, m_numberOfQueryUrls);
        m_urlGain1->Resize(1, m_numberOfQueryUrls);
        m_urlDiscount0->Resize(1, m_numberOfQueryUrls);
        m_urlDiscount1->Resize(1, m_numberOfQueryUrls);

        // keep one additional space to avoid pointer moving out
        m_urlSorter.resize(m_maxNumberOfUrlsPerQuery + 1);

        // prepared lookup table
        m_logWeights.resize(m_maxNumberOfUrlsPerQuery);
        size_t i = 0;
        for (typename std::vector<ElemType>::iterator it = m_logWeights.begin(); it != m_logWeights.end(); it++, i++)
        {
            *it = (ElemType)log(2.0 + i);
        }
    }

    virtual void CopyTo(ComputationNodeBasePtr nodeP, const std::wstring& newName, const CopyNodeFlags flags) const override
    {
        Base::CopyTo(nodeP, newName, flags);
        if (flags & CopyNodeFlags::copyNodeValue)
        {
            auto node = dynamic_pointer_cast<NDCG1EvalNode<ElemType>>(nodeP);
            node->m_urlGain0->SetValue(*m_urlGain0);
            node->m_urlGain1->SetValue(*m_urlGain1);
            node->m_urlDiscount0->SetValue(*m_urlDiscount0);
            node->m_urlDiscount1->SetValue(*m_urlDiscount1);

            node->m_queryUrls = m_queryUrls;
            node->m_urlSorter = m_urlSorter;
            node->m_logWeights = m_logWeights;
        }
    }

    // request matrices needed to do node function value evaluation
    virtual void RequestMatricesBeforeForwardProp(MatrixPool& matrixPool)
    {
        Base::RequestMatricesBeforeForwardProp(matrixPool);
        RequestMatrixFromPool(m_urlGain0, matrixPool);
        RequestMatrixFromPool(m_urlGain1, matrixPool);
        RequestMatrixFromPool(m_urlDiscount0, matrixPool);
        RequestMatrixFromPool(m_urlDiscount1, matrixPool);
    }

    // release gradient and temp matrices that no longer needed after all the children's gradients are computed.
    virtual void ReleaseMatricesAfterBackprop(MatrixPool& matrixPool)
    {
        Base::ReleaseMatricesAfterBackprop(matrixPool);
        ReleaseMatrixToPool(m_urlGain0, matrixPool);
        ReleaseMatrixToPool(m_urlGain1, matrixPool);
        ReleaseMatrixToPool(m_urlDiscount0, matrixPool);
        ReleaseMatrixToPool(m_urlDiscount1, matrixPool);

        // is this the right place?  it was not called after bp.
        m_queryUrls.clear();
        m_urlSorter.clear();
        m_logWeights.clear();
    }

protected:

    void UpdateCounts()
    {
        FrameRange fr(Input(0)->GetMBLayout());
        const Matrix<ElemType>& gains = Input(0)->ValueFor(fr);
        const Matrix<ElemType>& queryIds = Input(2)->ValueFor(fr);
        const size_t numberOfQueryUrls = gains.GetNumCols();
        int previousQueryId = -1;

        // Number of urls we have seen for the current query
        size_t numberOfUrls = 0;
        size_t maxNumberOfUrlsPerQuery = 0;
        for (size_t i = 0; i < numberOfQueryUrls; i++)
        {
            int queryId = (int)queryIds(0, i);
            if (queryId != previousQueryId)
            {
                if (numberOfUrls > maxNumberOfUrlsPerQuery)
                {
                    maxNumberOfUrlsPerQuery = numberOfUrls;
                }

                // New query
                numberOfUrls = 0;
                previousQueryId = queryId;
            }
            
            numberOfUrls++;
        }

        // Add last query.
        {
            if (numberOfUrls > maxNumberOfUrlsPerQuery)
            {
                maxNumberOfUrlsPerQuery = numberOfUrls;
            }
        }

        m_numberOfQueryUrls = numberOfQueryUrls;
        m_maxNumberOfUrlsPerQuery = maxNumberOfUrlsPerQuery;
    }

    struct Url
    {
        Url()
        {
            m_id = 0;
            m_rank0 = 0;
            m_rank = 0;
            m_score = (ElemType)0;
            m_gain = (ElemType)0;
        }

        Url(int _id, int _rk0, ElemType _sc, ElemType _gn) : m_id(_id), m_rank0(_rk0), m_rank(0), m_score(_sc), m_gain(_gn) {}

        int m_id;         // sample id
        int m_rank0;        // original rank based on label
        int m_rank;         // rank based on s in the associated query
        ElemType m_score;    // score
        ElemType m_gain;    // gain
        bool operator < (const Url &url) const{
            // tie breaking
            if (m_score == url.m_score || std::isnan(m_score) || std::isnan(url.m_score))
            {
                return m_gain < url.m_gain;
            }

            return m_score > url.m_score;
        }
    };

    struct QueryUrls
    {
        std::vector<Url> m_urls;
    };

    // master data structure
    std::list<QueryUrls> m_queryUrls;
    // buffer for sorting
    std::vector<Url> m_urlSorter;
    // lookup table for position based weights
    std::vector<ElemType> m_logWeights;

    size_t m_numberOfQueryUrls;
    size_t m_maxNumberOfUrlsPerQuery;
    // store the gains and weights
    shared_ptr<Matrix<ElemType>> m_urlGain0;
    shared_ptr<Matrix<ElemType>> m_urlGain1;
    shared_ptr<Matrix<ElemType>> m_urlDiscount0;
    shared_ptr<Matrix<ElemType>> m_urlDiscount1;
};

template class NDCG1EvalNode<float>;
template class NDCG1EvalNode<double>;

enum CTCDecodeType
{
    BestPath,
    PrefixSearch,
    BeamSearch,
    NotDecode
};



// Edit distance error evaluation node with the option of specifying penalty of substitution, deletion and insertion, as well as squashing the input sequences and ignoring certain samples.
// Using the classic DP algorithm as described in https://en.wikipedia.org/wiki/Edit_distance, adjusted to take into account the penalties.
// 
// The node allows to squash sequences of repeating labels and ignore certain labels. For example, if squashInputs is true and tokensToIgnore contains index of label '-' then
// given first input sequence as s1="a-ab-" and second as s2="-aa--abb" the edit distance will be computed against s1' = "aab" and s2' = "aab".
//
// The returned error is computed as: EditDistance(s1,s2) * length(s1') / length(s1)
//
// Just like ClassificationError and other evaluation nodes, when used as an evaluation criterion, the SGD process will aggregate all values over an epoch and report the average, i.e. the error rate.
// Primary objective of this node is for error evaluation of CTC training, see formula (1) in "Connectionist Temporal Classification: Labelling Unsegmented
// Sequence Data with Recurrent Neural Networks", ftp://ftp.idsia.ch/pub/juergen/icml2006.pdf
template<class ElemType>
class EditDistanceErrorNode : public ComputationNodeNonLooping/*ComputationNode*/<ElemType>, public NumInputs<2>
{
    typedef ComputationNodeNonLooping<ElemType> Base; UsingComputationNodeMembersBoilerplate;
    static const std::wstring TypeName() { return L"EditDistanceError"; }

public:
    // subPen - substitution penalty
    // delPen - deletion penalty
    // insPen - insertion penalty
    // squashInputs - whether to merge sequences of identical samples.
    // tokensToIgnore - list of indices of samples to ignore during edit distance evaluation
    EditDistanceErrorNode(DEVICEID_TYPE deviceId, const wstring & name, float subPen = 1.0f, float delPen = 1.0f, float insPen = 1.0f, bool squashInputs = false, vector<size_t> tokensToIgnore = {})
        : Base(deviceId, name), m_subPen(subPen), m_delPen(delPen), m_insPen(insPen), m_squashInputs(squashInputs), m_tokensToIgnore(tokensToIgnore)
    {
        m_tokensToIgnore.clear();
        m_tokensToIgnore.push_back(61);
        m_decodeType = CTCDecodeType::BestPath;
    }

    EditDistanceErrorNode(const ScriptableObjects::IConfigRecordPtr configp)
        : EditDistanceErrorNode(configp->Get(L"deviceId"), L"<placeholder>", configp->Get(L"subPen"), configp->Get(L"delPen"), configp->Get(L"insPen"), configp->Get(L"squashInputs"), {})
    {
        AttachInputsFromConfig(configp, this->GetExpectedNumInputs());
    }

    struct ArrayPool
    {
        vector<long double*>ptrs;

        long double* requestArray(int len, int& ptrId)
        {
            long double* ptr = new long double[len];
            ptrId = ptrs.size();
            ptrs.push_back(ptr);
            return ptr;
        }

        void releaseArray()
        {
            for (size_t i(0); i < ptrs.size(); ++i)
            {
                if (ptrs[i] != NULL)
                    delete[] ptrs[i];
            }
        }
    };

    struct BeamPath
    {
        vector<size_t>path;
        long double pb;
        long double pn;
        long double probability;
        BeamPath() {}

        bool friend operator<(const BeamPath& a, const BeamPath& b)
        {
            return a.probability > b.probability;
        }
    };

    struct PrefixPath
    {
        vector<unsigned short>path;
        long double* pn;
        long double* pb;
        long double probability;
        long double remainingProbability;
        int ptrId1;
        int ptrId2;

        PrefixPath() {}

        PrefixPath(int t, ArrayPool& ap)
        {
            pn = ap.requestArray(t, ptrId1);
            pb = ap.requestArray(t, ptrId2);
        }

        void release(ArrayPool& ap)
        {
            delete[] pn;
            delete[] pb;
            ap.ptrs[ptrId1] = NULL;
            ap.ptrs[ptrId2] = NULL;
        }

        bool friend operator<(const PrefixPath& a, const PrefixPath& b)
        {
            return a.probability < b.probability;
        }
    };

    void prefixSearchImpl(vector<vector<long double>>& CTC_probability, vector<size_t>& path)
    {
        ArrayPool ap;
        priority_queue<PrefixPath>pathQueue;
        int timeStep = CTC_probability.size();
        PrefixPath bestPath(timeStep, ap);
        PrefixPath result;
        bestPath.probability = (long double)1.0;
        for (int i(0); i < timeStep; ++i)
        {
            bestPath.probability *= CTC_probability[i][40];
            bestPath.pb[i] = bestPath.probability;
            bestPath.pn[i] = (long double)0.0;
        }
        bestPath.remainingProbability = 1 - bestPath.probability;
        result.probability = bestPath.probability;

        while (bestPath.remainingProbability > result.probability)
        {
            long double remainingProbability = bestPath.remainingProbability;
            int pathLen = bestPath.path.size();
            for (unsigned short i(0); i < 40; ++i)
            {
                PrefixPath extendedPath(timeStep, ap);
                extendedPath.path.resize(pathLen + 1);
                for (int j(0); j < pathLen; ++j)
                    extendedPath.path[j] = bestPath.path[j];
                extendedPath.path[pathLen] = i;
                if (0 == pathLen)
                    extendedPath.pn[0] = CTC_probability[0][i];
                else
                    extendedPath.pn[0] = (long double)0.0;
                extendedPath.pb[0] = (long double)0.0;

                long double prefixProb = extendedPath.pn[0];
                for (int j(1); j < timeStep; ++j)
                {
                    long double newLabelProb = bestPath.pb[j - 1];
                    if (0 == pathLen || bestPath.path[pathLen - 1] != i)
                        newLabelProb += bestPath.pn[j - 1];
                    extendedPath.pn[j] = CTC_probability[j][i] * (newLabelProb + extendedPath.pn[j - 1]);
                    extendedPath.pb[j] = CTC_probability[j][40] * (extendedPath.pb[j - 1] + extendedPath.pn[j - 1]);
                    prefixProb += CTC_probability[j][i] * newLabelProb;
                }

                extendedPath.probability = extendedPath.pn[timeStep - 1] + extendedPath.pb[timeStep - 1];
                extendedPath.remainingProbability = prefixProb - extendedPath.probability;
                remainingProbability -= extendedPath.remainingProbability;

                if (extendedPath.probability > result.probability)
                    result = extendedPath;
                if (extendedPath.remainingProbability > result.probability)
                    pathQueue.push(extendedPath);
                else
                    extendedPath.release(ap);
                if (remainingProbability <= result.probability)
                    break;
            }
            bestPath.release(ap);

            if (pathQueue.empty())
                break;
            bestPath = pathQueue.top();
            pathQueue.pop();
        }

        ap.releaseArray();

        for (size_t i(0); i < result.path.size(); ++i)
        {
            if (result.path[i] != 39)
                path.push_back((size_t)(result.path[i]));
        }
    }

    void beamSearchImpl(const vector<vector<long double>>& CTC_probability, vector<size_t>& path)
    {
        int timeStep = CTC_probability.size();
        vector<BeamPath>bestBeamPaths;
        vector<BeamPath>tempPaths;
        for (size_t i(0); i < 40; ++i)
        {
            BeamPath beamPath;
            beamPath.path.push_back(i);
            beamPath.pb = (long double)0.0;
            beamPath.pn = CTC_probability[0][i];
            beamPath.probability = CTC_probability[0][i];
            bestBeamPaths.push_back(beamPath);
        }
        BeamPath beamPath;
        beamPath.pn = (long double)0.0;
        beamPath.pb = CTC_probability[0][40];
        beamPath.probability = CTC_probability[0][40];
        bestBeamPaths.push_back(beamPath);
        sort(bestBeamPaths.begin(), bestBeamPaths.end());


        for (int i(1); i < timeStep; ++i)
        {
            tempPaths.clear();
            for (int j(0); j < m_beamWidth && j < bestBeamPaths.size(); ++j)
            {
                BeamPath p = bestBeamPaths[j];
                int len = p.path.size();

                BeamPath tempPath0;
                tempPath0.path.resize(len);
                for (int k(0); k < len; ++k)
                    tempPath0.path[k] = p.path[k];
                if (len != 0)
                {
                    tempPath0.pn = p.pn * CTC_probability[i][p.path[len - 1]];
                    for (size_t k(0); k < bestBeamPaths.size(); ++k)
                    {
                        if (bestBeamPaths[k].path.size() == len - 1)
                        {
                            bool flag = true;
                            for (size_t l(0); l < len - 1; ++l)
                            {
                                if (bestBeamPaths[k].path[l] != tempPath0.path[l])
                                {
                                    flag = false;
                                    break;
                                }
                            }
                            if (flag)
                            {
                                tempPath0.pn += bestBeamPaths[k].pb * CTC_probability[i][p.path[len - 1]];
                                if (1 == len || p.path[len - 1] != p.path[len - 2])
                                    tempPath0.pn += bestBeamPaths[k].pn * CTC_probability[i][p.path[len - 1]];
                                break;
                            }
                        }
                    }
                }
                tempPath0.pb = p.probability * CTC_probability[i][40];
                tempPath0.probability = tempPath0.pb + tempPath0.pn;
                tempPaths.push_back(tempPath0);

                for (size_t k(0); k < 40; ++k)
                {
                    BeamPath tempPath;
                    tempPath.path.resize(len + 1);
                    for (int l(0); l < len; ++l)
                        tempPath.path[l] = p.path[l];
                    tempPath.path[len] = k;
                    tempPath.pb = (long double)0.0;
                    tempPath.pn = p.pb * CTC_probability[i][k];
                    if (0 == len || k != p.path[len - 1])
                        tempPath.pn += p.pn * CTC_probability[i][k];
                    tempPath.probability = tempPath.pn;
                    tempPaths.push_back(tempPath);
                }
            }

            bestBeamPaths.clear();
            sort(tempPaths.begin(), tempPaths.end());
            for (int j(0); j < m_beamWidth && j < tempPaths.size(); ++j)
                bestBeamPaths.push_back(tempPaths[j]);
        }

        for (size_t i(0); i < bestBeamPaths[0].path.size(); ++i)
        {
            if (bestBeamPaths[0].path[i] != 39)
                path.push_back(bestBeamPaths[0].path[i]);
        }
    }

    // For Timit, 41 labels (40 labels + 1 blank)
    ElemType calPathProbabilty(const vector<size_t>& label, const vector<vector<long double>>& probabilty)
    {
        int timeStep = probabilty.size();
        int length = label.size() * 2 + 1;
        vector<vector<long double>>alpha;
        alpha.resize(timeStep);
        for (size_t i(0); i < alpha.size(); ++i)
        {
            alpha[i].resize(length);
            fill(alpha[i].begin(), alpha[i].end(), 0);
        }

        //double result = log(probabilty[0][40] + probabilty[0][label[0]]);

        alpha[0][0] = probabilty[0][40];// / (probabilty[0][40] + probabilty[0][label[0]]);
        alpha[0][1] = probabilty[0][label[0]];// / (probabilty[0][40] + probabilty[0][label[0]]);

        for (int i(1); i < timeStep; ++i)
        {
            alpha[i][0] = alpha[i - 1][0] * probabilty[i][40];
            alpha[i][1] = (alpha[i - 1][0] + alpha[i - 1][1]) * probabilty[i][label[0]];
            for (int j(2); j < length; ++j)
            {
                if (j >= (i + 1) * 2)
                    break;

                double _alpha = alpha[i - 1][j] + alpha[i - 1][j - 1];
                if ((!(j & 1)) || label[j >> 1] == label[(j >> 1) - 1])
                {
                    if (j & 1)
                        alpha[i][j] = _alpha * probabilty[i][label[j >> 1]];
                    else
                        alpha[i][j] = _alpha * probabilty[i][40];
                }
                else
                {
                    if (j & 1)
                        alpha[i][j] = (_alpha + alpha[i - 1][j - 2]) * probabilty[i][label[j >> 1]];
                    else
                        alpha[i][j] = (_alpha + alpha[i - 1][j - 2]) * probabilty[i][40];
                }
            }

            /*
            double sum = 0;
            for (size_t j(0); j < length; ++j)
            sum += alpha[i][j];
            for (size_t j(0); j < length; ++j)
            alpha[i][j] /= sum;
            result += log(sum);*/
        }

        return alpha[timeStep - 1][length - 1] + alpha[timeStep - 1][length - 2];
    }

    // firstSeq - first sequence of samples
    // secondSeq - second sequence of samples
    // numParallelSequences - number of parallel sequences in the minibatch
    // subPen - substitution penalty
    // delPen - deletion penalty
    // insPen - insertion penalty
    // squashInputs - whether to merge sequences of identical samples.
    // tokensToIgnore - list of samples to ignore during edit distance evaluation
    ElemType TIMIT_ComputeEditDistanceError(const CTCDecodeType decodeType, const vector<ElemType*>& labelPtr, const vector<ElemType*>& outputPtr, const vector<size_t>& colNum, const size_t rowNum,
        float subPen, float delPen, float insPen, bool squashInputs, const size_t tokensToIgnore = 61)
    {
        ElemType result = 0;
        float del, ins, sub;
        Matrix<float>grid(CPUDEVICE); // Edit distance between subsequences
        Matrix<float>insMatrix(CPUDEVICE); // Number of insertions between subsequences
        Matrix<float>delMatrix(CPUDEVICE); //Number of deletions between subsequences
        Matrix<float>subMatrix(CPUDEVICE); // Number of substitutions between subsequences

        for (size_t sequenceId(0); sequenceId < colNum.size(); ++sequenceId)
        {
            vector<size_t>v1;
            vector<size_t>v2;
            size_t pre = SIZE_MAX;
            for (size_t i(0); i < colNum[sequenceId]; ++i)
            {
                if ((size_t)labelPtr[sequenceId][i] == 46)
                    pre = SIZE_MAX;
                else if (TIMIT_61to39_map[(size_t)labelPtr[sequenceId][i]] != pre)
                {
                    v1.push_back(TIMIT_61to39_map[(size_t)labelPtr[sequenceId][i]]);
                    v1.push_back(TIMIT_61to39_map[(size_t)labelPtr[sequenceId][i]]);
                    v1.push_back(TIMIT_61to39_map[(size_t)labelPtr[sequenceId][i]]);
                    pre = TIMIT_61to39_map[(size_t)labelPtr[sequenceId][i]];
                }
            }

            if (CTCDecodeType::BestPath == decodeType)
            {
                size_t index = 0;
                pre = SIZE_MAX;
                for (size_t i(0); i < colNum[sequenceId]; ++i)
                {
                    size_t maxOutputId = 0;
                    ElemType maxOutput = outputPtr[sequenceId][index++];
                    for (size_t j(1); j < rowNum; ++j)
                    {
                        if (outputPtr[sequenceId][index] > maxOutput)
                        {
                            maxOutput = outputPtr[sequenceId][index];
                            maxOutputId = j;
                        }
                        ++index;
                    }
                    if (maxOutputId == 46)
                    {
                        pre = SIZE_MAX;
                        continue;
                    }
                    if (maxOutputId == tokensToIgnore)
                        pre = SIZE_MAX;
                    else if (TIMIT_61to39_map[maxOutputId] != pre)
                    {
                        v2.push_back(TIMIT_61to39_map[maxOutputId]);
                        pre = TIMIT_61to39_map[maxOutputId];
                    }
                }
            }
            else if (CTCDecodeType::PrefixSearch == decodeType)
            {
                long double* probability = new long double[rowNum * colNum[sequenceId]];
                size_t index = 0;
                for (size_t i(0); i < colNum[sequenceId]; ++i)
                {
                    long double sum = 0;
                    long double maxOutput = outputPtr[sequenceId][index];
                    for (size_t j(1); j < rowNum; ++j)
                    {
                        if (outputPtr[sequenceId][index + j] > maxOutput)
                            maxOutput = outputPtr[sequenceId][index + j];
                    }
                    for (size_t j(0); j < rowNum; ++j)
                    {
                        probability[index] = exp(outputPtr[sequenceId][index] - maxOutput);
                        sum += probability[index++];
                    }
                    index -= rowNum;
                    for (size_t j(0); j < rowNum; ++j)
                        probability[index++] /= sum;
                }

                // Step 1 : find out Sections and pack data
                vector<size_t>sectionBegin;
                vector<size_t>sectionEnd;
                bool flag = false;
                for (size_t i(0); i < colNum[sequenceId]; ++i)
                {
                    if (probability[i * rowNum + 61] > m_threshold)
                    {
                        if (flag)
                        {
                            sectionEnd.push_back(i - 1);
                            flag = false;
                        }
                    }
                    else
                    {
                        if (!flag)
                        {
                            sectionBegin.push_back(i);
                            flag = true;
                        }
                    }
                }
                if (flag)
                    sectionEnd.push_back(colNum[sequenceId] - 1);

                // Step 2 : calculate path for each section
                for (size_t i(0); i < sectionBegin.size(); ++i)
                {
                    vector<vector<long double>>CTC_probability;
                    CTC_probability.resize(sectionEnd[i] - sectionBegin[i] + 1);
                    index = sectionBegin[i] * rowNum;
                    for (size_t j(0); j < sectionEnd[i] - sectionBegin[i] + 1; ++j)
                    {
                        CTC_probability[j].resize(41);
                        fill(CTC_probability[j].begin(), CTC_probability[j].end(), 0);
                        for (size_t k(0); k < 62; ++k)
                            CTC_probability[j][TIMIT_61to39_map[k]] += probability[index++];
                    }

                    prefixSearchImpl(CTC_probability, v2);
                }

                delete[] probability;
            }
            else if (CTCDecodeType::BeamSearch == decodeType)
            {
                long double* probability = new long double[rowNum * colNum[sequenceId]];
                size_t index = 0;
                for (size_t i(0); i < colNum[sequenceId]; ++i)
                {
                    long double sum = 0;
                    long double maxOutput = outputPtr[sequenceId][index];
                    for (size_t j(1); j < rowNum; ++j)
                    {
                        if (outputPtr[sequenceId][index + j] > maxOutput)
                            maxOutput = outputPtr[sequenceId][index + j];
                    }
                    for (size_t j(0); j < rowNum; ++j)
                    {
                        probability[index] = exp(outputPtr[sequenceId][index] - maxOutput);
                        sum += probability[index++];
                    }
                    index -= rowNum;
                    for (size_t j(0); j < rowNum; ++j)
                        probability[index++] /= sum;
                }

                vector<vector<long double>>CTC_probability;
                CTC_probability.resize(colNum[sequenceId]);
                index = 0;
                for (size_t i(0); i < colNum[sequenceId]; ++i)
                {
                    CTC_probability[i].resize(41);
                    fill(CTC_probability[i].begin(), CTC_probability[i].end(), (long double)0.0);
                    for (size_t j(0); j < 62; ++j)
                        CTC_probability[i][TIMIT_61to39_map[j]] += probability[index++];
                }
                beamSearchImpl(CTC_probability, v2);

                delete[] probability;
            }
            else
                LogicError("TIMIT_ComputeEditDistanceError : unkowned decodeType.");

            size_t len1 = v1.size();
            size_t len2 = v2.size();
            grid.Resize(len1 + 1, len2 + 1);
            insMatrix.Resize(len1 + 1, len2 + 1);
            delMatrix.Resize(len1 + 1, len2 + 1);
            subMatrix.Resize(len1 + 1, len2 + 1);
            insMatrix.SetValue(0.0f);
            delMatrix.SetValue(0.0f);
            subMatrix.SetValue(0.0f);

            for (size_t i = 0; i < len1 + 1; i++)
            {
                grid(i, 0) = (float)(i * delPen);
                delMatrix(i, 0) = (float)i;
            }

            for (size_t j = 0; j < len2 + 1; j++)
            {
                grid(0, j) = (float)(j * insPen);
                insMatrix(0, j) = (float)j;
            }
            for (size_t i = 1; i < len1 + 1; i++)
            {
                for (size_t j = 1; j < len2 + 1; j++)
                {
                    if (v1[i - 1] == v2[j - 1])
                    {
                        grid(i, j) = grid(i - 1, j - 1);
                        insMatrix(i, j) = insMatrix(i - 1, j - 1);
                        delMatrix(i, j) = delMatrix(i - 1, j - 1);
                        subMatrix(i, j) = subMatrix(i - 1, j - 1);
                    }
                    else
                    {
                        del = grid(i - 1, j) + delPen; //deletion 
                        ins = grid(i, j - 1) + insPen;  //insertion
                        sub = grid(i - 1, j - 1) + subPen; //substitution 
                        if (sub <= del && sub <= ins)
                        {
                            insMatrix(i, j) = insMatrix(i - 1, j - 1);
                            delMatrix(i, j) = delMatrix(i - 1, j - 1);
                            subMatrix(i, j) = subMatrix(i - 1, j - 1) + 1.0f;
                            grid(i, j) = sub;
                        }
                        else if (del < ins)
                        {
                            insMatrix(i, j) = insMatrix(i - 1, j);
                            subMatrix(i, j) = subMatrix(i - 1, j);
                            delMatrix(i, j) = delMatrix(i - 1, j) + 1.0f;
                            grid(i, j) = del;
                        }
                        else
                        {
                            delMatrix(i, j) = delMatrix(i, j - 1);
                            subMatrix(i, j) = subMatrix(i, j - 1);
                            insMatrix(i, j) = insMatrix(i, j - 1) + 1.0f;
                            grid(i, j) = ins;
                        }
                    }
                }
            }

            result += (insMatrix(len1, len2) + delMatrix(len1, len2) + subMatrix(len1, len2)) * colNum[sequenceId] / len1;
        }

        return result;
    }

    virtual void BackpropToNonLooping(size_t /*inputIndex*/) override
    {
        LogicError("%ls operation is used for evaluation only.", OperationName().c_str());
    }

    virtual void ForwardPropNonLooping() override
    {
        bool isInput0Sparse = Input(0)->template Is<SparseInputValue<ElemType>>();
        bool isInput1Sparse = Input(1)->template Is<SparseInputValue<ElemType>>();
        if (isInput0Sparse || isInput1Sparse)
            LogicError("EditDistanceError node was not tested for sparse inputs.");

        FrameRange frameRange(Input(0)->GetMBLayout());
        Input(0)->ValueFor(frameRange).VectorMax(*m_maxIndexes0, *m_maxValues, true);
        Input(1)->ValueFor(frameRange).VectorMax(*m_maxIndexes1, *m_maxValues, true);

        MaskMissingColumnsToZero(*m_maxIndexes0, Input(0)->GetMBLayout(), frameRange);
        MaskMissingColumnsToZero(*m_maxIndexes1, Input(1)->GetMBLayout(), frameRange);
        if (Environment().IsTraining())
            Value()(0, 0) = ComputeEditDistanceError(*m_maxIndexes0, *m_maxIndexes1, Input(0)->GetMBLayout(), m_subPen, m_delPen, m_insPen, m_squashInputs, m_tokensToIgnore);
        else
        {
            if (CTCDecodeType::NotDecode == m_decodeType)
                Value()(0, 0) = ComputeEditDistanceError(*m_maxIndexes0, *m_maxIndexes1, Input(0)->GetMBLayout(), m_subPen, m_delPen, m_insPen, m_squashInputs, m_tokensToIgnore);
            else
            {
                ElemType* _labelPtr = Input(0)->ValueFor(frameRange).CopyToArray();
                ElemType* _outputPtr = Input(1)->ValueFor(frameRange).CopyToArray();
                size_t row = Input(0)->ValueFor(frameRange).GetNumRows();
                if (row != 62)
                    LogicError("Row %d not match 62.", row);
                vector<ElemType*>labelPtr;
                vector<ElemType*>outputPtr;
                vector<size_t>colNum;

                for (const auto& sequence : Input(0)->GetMBLayout()->GetAllSequences())
                {
                    if (sequence.seqId == GAP_SEQUENCE_ID)
                        continue;

                    auto numFrames = Input(0)->GetMBLayout()->GetNumSequenceFramesInCurrentMB(sequence);

                    if (numFrames > 0)
                    {
                        auto columnIndices = Input(0)->GetMBLayout()->GetColumnIndices(sequence);
                        size_t columnIndicesNum = columnIndices.size();
                        colNum.push_back(columnIndicesNum);
                        ElemType* labelPtr_ = new ElemType[columnIndicesNum];
                        ElemType* outputPtr_ = new ElemType[columnIndicesNum * row];
                        size_t labelIndex = 0;
                        size_t outputIndex = 0;

                        for (size_t i(0); i < columnIndicesNum; ++i)
                        {
                            size_t index = columnIndices[i] * row;
                            ElemType maxLabel = _labelPtr[index];
                            size_t maxLabelId = 0;
                            outputPtr_[outputIndex++] = _outputPtr[index++];

                            for (size_t j(1); j < row; ++j)
                            {
                                if (_labelPtr[index] > maxLabel)
                                {
                                    maxLabel = _labelPtr[index];
                                    maxLabelId = j;
                                }
                                outputPtr_[outputIndex++] = _outputPtr[index++];
                            }
                            labelPtr_[labelIndex++] = maxLabelId;
                        }

                        labelPtr.push_back(labelPtr_);
                        outputPtr.push_back(outputPtr_);
                    }
                }
                delete[] _labelPtr;
                delete[] _outputPtr;

                Value()(0, 0) = TIMIT_ComputeEditDistanceError(m_decodeType, labelPtr, outputPtr, colNum, row, m_subPen, m_delPen, m_insPen, m_squashInputs, m_tokensToIgnore[0]);

                for (size_t i(0); i < labelPtr.size(); ++i)
                    delete[] labelPtr[i];
                for (size_t i(0); i < outputPtr.size(); ++i)
                    delete[] outputPtr[i];
            }
        }
        Value().TransferToDeviceIfNotThere(Input(0)->GetDeviceId());
    }

    virtual void Validate(bool isFinalValidationPass) override
    {
        ValidateBinaryReduce(isFinalValidationPass);
    }

    virtual void UpdateFunctionMBSize() override
    {
        Base::UpdateFunctionMBSize();

        // resize the temporaries to their proper size
        size_t cols = Input(0)->Value().GetNumCols();
        m_maxIndexes0->Resize(1, cols);
        m_maxIndexes1->Resize(1, cols);
        m_maxValues->Resize(1, cols);
    }

    virtual void CopyTo(ComputationNodeBasePtr  nodeP, const std::wstring& newName, const CopyNodeFlags flags) const override
    {
        Base::CopyTo(nodeP, newName, flags);

        if (flags & CopyNodeFlags::copyNodeValue)
        {
            auto node = dynamic_pointer_cast<EditDistanceErrorNode<ElemType>>(nodeP);
            node->m_maxIndexes0 = m_maxIndexes0;
            node->m_maxIndexes1 = m_maxIndexes1;
            node->m_maxValues = m_maxValues;
            node->m_squashInputs = m_squashInputs;
            node->m_subPen = m_subPen;
            node->m_delPen = m_delPen;
            node->m_insPen = m_insPen;
            node->m_tokensToIgnore = m_tokensToIgnore;
            node->m_decodeType = m_decodeType;
        }
    }

    //request matrices needed to do node function value evaluation
    virtual void RequestMatricesBeforeForwardProp(MatrixPool& matrixPool)
    {
        Base::RequestMatricesBeforeForwardProp(matrixPool);
        RequestMatrixFromPool(m_maxIndexes0, matrixPool);
        RequestMatrixFromPool(m_maxIndexes1, matrixPool);
        RequestMatrixFromPool(m_maxValues, matrixPool);
    }

    //release temp matrices that are only used by forward computation
    //don't release matrices that need to be used in the gradient computation
    virtual void ReleaseMatricesAfterForwardProp(MatrixPool& matrixPool)
    {
        Base::ReleaseMatricesAfterForwardProp(matrixPool);
        ReleaseMatrixToPool(m_maxIndexes0, matrixPool);
        ReleaseMatrixToPool(m_maxIndexes1, matrixPool);
        ReleaseMatrixToPool(m_maxValues, matrixPool);
    }

    // firstSeq - first sequence of samples
    // secondSeq - second sequence of samples
    // numParallelSequences - number of parallel sequences in the minibatch
    // subPen - substitution penalty
    // delPen - deletion penalty
    // insPen - insertion penalty
    // squashInputs - whether to merge sequences of identical samples.
    // tokensToIgnore - list of samples to ignore during edit distance evaluation
    ElemType ComputeEditDistanceError(Matrix<ElemType>& firstSeq, const Matrix<ElemType> & secondSeq, MBLayoutPtr pMBLayout,
        float subPen, float delPen, float insPen, bool squashInputs, const vector<size_t>& tokensToIgnore)
    {
        std::vector<int> firstSeqVec, secondSeqVec;

        // Edit distance between subsequences
        Matrix<float> grid(CPUDEVICE);

        // Number of insertions between subsequences
        Matrix<float> insMatrix(CPUDEVICE);

        //Number of deletions between subsequences
        Matrix<float> delMatrix(CPUDEVICE);

        // Number of substitutions between subsequences
        Matrix<float> subMatrix(CPUDEVICE);

        float del, ins, sub;
        ElemType wrongSampleNum = 0.0;
        size_t totalSampleNum = 0, totalframeNum = 0;
        size_t sequenceStartFrame = 0;

        for (const auto& sequence : pMBLayout->GetAllSequences())
        {
            if (sequence.seqId == GAP_SEQUENCE_ID)
                continue;

            auto numFrames = pMBLayout->GetNumSequenceFramesInCurrentMB(sequence);

            if (numFrames > 0)
            {
                totalframeNum += numFrames;

                auto columnIndices = pMBLayout->GetColumnIndices(sequence);

                ExtractSampleSequence(firstSeq, columnIndices, squashInputs, tokensToIgnore, firstSeqVec);
                ExtractSampleSequence(secondSeq, columnIndices, squashInputs, tokensToIgnore, secondSeqVec);

                //calculate edit distance
                size_t firstSize = firstSeqVec.size();
                size_t secondSize = secondSeqVec.size();
                if (Base::HasEnvironmentPtr() && Base::Environment().IsV2Library())
                    totalSampleNum += secondSize;
                else
                    totalSampleNum += firstSize;

                grid.Resize(firstSize + 1, secondSize + 1);
                insMatrix.Resize(firstSize + 1, secondSize + 1);
                delMatrix.Resize(firstSize + 1, secondSize + 1);
                subMatrix.Resize(firstSize + 1, secondSize + 1);
                insMatrix.SetValue(0.0f);
                delMatrix.SetValue(0.0f);
                subMatrix.SetValue(0.0f);

                for (size_t i = 0; i < firstSize + 1; i++)
                {
                    grid(i, 0) = (float)(i * delPen);
                    delMatrix(i, 0) = (float)i;
                }

                for (size_t j = 0; j < secondSize + 1; j++)
                {
                    grid(0, j) = (float)(j * insPen);
                    insMatrix(0, j) = (float)j;
                }
                for (size_t i = 1; i < firstSize + 1; i++)
                {
                    for (size_t j = 1; j < secondSize + 1; j++)
                    {
                        if (firstSeqVec[i - 1] == secondSeqVec[j - 1])
                        {
                            grid(i, j) = grid(i - 1, j - 1);
                            insMatrix(i, j) = insMatrix(i - 1, j - 1);
                            delMatrix(i, j) = delMatrix(i - 1, j - 1);
                            subMatrix(i, j) = subMatrix(i - 1, j - 1);
                        }
                        else
                        {
                            del = grid(i - 1, j) + delPen; //deletion 
                            ins = grid(i, j - 1) + insPen;  //insertion
                            sub = grid(i - 1, j - 1) + subPen; //substitution 
                            if (sub <= del && sub <= ins)
                            {
                                insMatrix(i, j) = insMatrix(i - 1, j - 1);
                                delMatrix(i, j) = delMatrix(i - 1, j - 1);
                                subMatrix(i, j) = subMatrix(i - 1, j - 1) + 1.0f;
                                grid(i, j) = sub;
                            }
                            else if (del < ins)
                            {
                                insMatrix(i, j) = insMatrix(i - 1, j);
                                subMatrix(i, j) = subMatrix(i - 1, j);
                                delMatrix(i, j) = delMatrix(i - 1, j) + 1.0f;
                                grid(i, j) = del;
                            }
                            else
                            {
                                delMatrix(i, j) = delMatrix(i, j - 1);
                                subMatrix(i, j) = subMatrix(i, j - 1);
                                insMatrix(i, j) = insMatrix(i, j - 1) + 1.0f;
                                grid(i, j) = ins;
                            }
                        }
                    }
                }

                wrongSampleNum += insMatrix(firstSize, secondSize) + delMatrix(firstSize, secondSize) + subMatrix(firstSize, secondSize);
            }

            sequenceStartFrame += numFrames;
        }

        return (ElemType)(wrongSampleNum * totalframeNum / totalSampleNum);
    }

    virtual void Save(File& fstream) const override
    {
        Base::Save(fstream);
        fstream << m_subPen;
        fstream << m_delPen;
        fstream << m_insPen;
        fstream << m_squashInputs;
        fstream << m_tokensToIgnore;
        fstream << m_decodeType;
        fstream << m_threshold;
        fstream << m_beamWidth;
    }

    virtual void Load(File& fstream, size_t modelVersion) override
    {
        Base::Load(fstream, modelVersion);
        fstream >> m_subPen;
        fstream >> m_delPen;
        fstream >> m_insPen;
        fstream >> m_squashInputs;
        fstream >> m_tokensToIgnore;
        fstream >> m_decodeType;
        fstream >> m_threshold;
        fstream >> m_beamWidth;

        ifstream inFile("decodeType.txt", ios::in);
        if (inFile)
        {
            string decodeType;
            inFile >> decodeType;
            if (decodeType == "prefixsearch" || decodeType == "Prefixsearch" || decodeType == "prefixSearch" || decodeType == "PrefixSearch")
            {
                inFile >> m_threshold;
                if (m_threshold > 0)
                {
                    m_decodeType = CTCDecodeType::PrefixSearch;
                    cout << "Decode type : PrefixSearch\tthreshold = " << m_threshold << endl;
                }
                else
                    LogicError("In prefix search, threshold should be greater than 0.");
            }
            else if (decodeType == "beamsearch" || decodeType == "Beamsearch" || decodeType == "beamSearch" || decodeType == "BeamSearch")
            {
                inFile >> m_beamWidth;
                if (m_beamWidth > 0)
                {
                    m_decodeType = CTCDecodeType::BeamSearch;
                    cout << "Decode type : BeamSearch\tbeamWidth = " << m_beamWidth << endl;
                }
                else
                    LogicError("In beam search, beam width should be greater than 0.");
            }
            else if (decodeType != "bestpath" && decodeType != "Bestpath" &&decodeType != "bestPath" &&decodeType != "BestPath")
                LogicError("Not support decode type : %s.", decodeType.c_str());
            else
                cout << "Decode type : BestPath" << endl;
        }
        else
            cout << "Decode type : BestPath" << endl;
        inFile.close();
    }

    float SubstitutionPenalty() const { return m_subPen; }
    float DeletionPenalty() const { return m_delPen; }
    float InsertionPenalty() const { return m_insPen; }
    bool SquashInputs() const { return m_squashInputs; }
    std::vector<size_t> TokensToIgnore() const { return m_tokensToIgnore; }

private:
    shared_ptr<Matrix<ElemType>> m_maxIndexes0, m_maxIndexes1;
    shared_ptr<Matrix<ElemType>> m_maxValues;
    bool m_squashInputs;
    float m_subPen;
    float m_delPen;
    float m_insPen;
    std::vector<size_t> m_tokensToIgnore;
    CTCDecodeType m_decodeType;
    long double m_threshold;
    int m_beamWidth;

    // Clear out_SampleSeqVec and extract a vector of samples from the matrix into out_SampleSeqVec.
    static void ExtractSampleSequence(const Matrix<ElemType>& firstSeq, vector<size_t>& columnIndices, bool squashInputs, const vector<size_t>& tokensToIgnore, std::vector<int>& out_SampleSeqVec)
    {
        out_SampleSeqVec.clear();

        // Get the first element in the sequence
        size_t lastId = (int)firstSeq(0, columnIndices[0]);
        if (std::find(tokensToIgnore.begin(), tokensToIgnore.end(), lastId) == tokensToIgnore.end())
            out_SampleSeqVec.push_back(lastId);

        // Remaining elements
        if (squashInputs)
        {
            //squash sequences of identical samples
            for (size_t i = 1; i < columnIndices.size(); i++)
            {
                size_t refId = (int)firstSeq(0, columnIndices[i]);
                if (lastId != refId)
                {
                    lastId = refId;
                    if (std::find(tokensToIgnore.begin(), tokensToIgnore.end(), refId) == tokensToIgnore.end())
                        out_SampleSeqVec.push_back(refId);
                }
            }
        }
        else
        {
            for (size_t i = 1; i < columnIndices.size(); i++)
            {
                auto refId = (int)firstSeq(0, columnIndices[i]);
                if (std::find(tokensToIgnore.begin(), tokensToIgnore.end(), refId) == tokensToIgnore.end())
                    out_SampleSeqVec.push_back(refId);
            }
        }
    }
};

template class EditDistanceErrorNode<float>;
template class EditDistanceErrorNode<double>;

// OneHotNode will create corresponding one hot tensor based on the input tensor. 
template <class ElemType>
class OneHotNode : public ComputationNodeNonLooping<ElemType>, public NumInputs<1>
{
    typedef ComputationNodeNonLooping<ElemType> Base;
    UsingComputationNodeMembersBoilerplate;
    static const std::wstring TypeName()
    {
        return L"OneHot";
    }

public:
    OneHotNode(DEVICEID_TYPE deviceId, size_t num_class, bool is_sparse, int axis, const wstring& name) : Base(deviceId, name)
    {
        m_num_class = num_class;
        m_sparse = is_sparse;
        m_axis = axis;
        m_offset = -1;
    }
    //do we really need this?
    OneHotNode(DEVICEID_TYPE deviceId, const wstring& name) : OneHotNode(deviceId, 0, false, -1, name)
    {
    }

    OneHotNode(const ScriptableObjects::IConfigRecordPtr configp)
        : OneHotNode(configp->Get(L"deviceId"), configp->Get(L"numClass"))
    {
        AttachInputsFromConfig(configp, this->GetExpectedNumInputs());
    }

    virtual void ForwardPropNonLooping() override
    {
        auto& dims = GetSampleLayout().GetDims();
        vector<size_t> shape;
        shape.assign(dims.begin(), dims.end());

        if (m_offset < 0)
        {
            CalculateAxisOffset();
        }

        auto& output = Value();
        output.AssignOneHot(InputRef(0).Value(), shape, m_offset, m_sparse);
    }

    virtual void BackpropToNonLooping(size_t inputIndex) override
    {
        LogicError("The node \"%ls\" can be used in training, but it does not participate in gradient propagation.", OperationName().c_str());
    }

    virtual bool OutputUsedInComputingInputNodesGradients() const override {
        return false;
    }

    virtual bool InputUsedInComputingInputNodesGradients(size_t /*childIndex*/) const override {
        return false;
    }

    virtual void /*ComputationNodeBase::*/ Validate(bool isFinalValidationPass) override
    {
        Base::Validate(isFinalValidationPass);

        Base::m_isValueSparse = m_sparse;
        if (m_offset < 0)
        {
            CalculateAxisOffset();
        }

        const auto& inputSampleLayout = Input(0)->GetSampleLayout();
        const auto& inputDims = inputSampleLayout.GetDims();
        SmallVector<size_t> dims;
        if (m_offset > 0)
        {
            dims.append(inputDims.begin(), inputDims.begin() + m_offset);
        }
        dims.push_back(m_num_class);
        if (m_offset != inputDims.size())
        {
            dims.append(inputDims.begin() + m_offset, inputDims.end());
        }

        auto sampleLayout = TensorShape(dims);
        m_pMBLayout = Input(0)->GetMBLayout();

        SetDims(sampleLayout, HasMBLayout());
    }

protected:
    void CalculateAxisOffset()
    {
        if (m_offset < 0)
        {
            const auto& inputSampleLayout = Input(0)->GetSampleLayout();
            const auto& inputDims = inputSampleLayout.GetDims();
            size_t len = inputDims.size();
            m_offset = m_axis < 0 ? (len + 1 + m_axis) % (len + 1) : m_axis % (len + 1);
        }
    }

    size_t m_num_class;
    bool m_sparse;
    int m_axis;
    int m_offset;
};

template class OneHotNode<float>;
template class OneHotNode<double>;

#ifdef COMING_SOON

// -----------------------------------------------------------------------
// SequenceDecoderNode (label, position_dependent_score, transition_score)
// Decoder that matches CRF training.
//  - label : output label vector of [0:T-1]
//  - position_dependent_score : score from position dependent node,
//    in the R-CRF case, it is the RNN output score before softmax
//  - transition score : score from the transition node,
//    in the R-CRF case, it is the transition probability between labels
// -----------------------------------------------------------------------

template <class ElemType>
class SequenceDecoderNode : public ComputationNodeNonLooping /*ComputationNode*/<ElemType>, public NumInputs<3>
{
    typedef ComputationNodeNonLooping<ElemType> Base;
    UsingComputationNodeMembersBoilerplate;
    static const std::wstring TypeName()
    {
        return L"SequenceDecoderNode";
    }

private:
    // TODO: member variables go to the end
    Matrix<ElemType> mAlpha;
    Matrix<ElemType> mBacktrace;

    int mStartLab; // the starting output label
    int mEndLab;   // the ending output label, if available
    ElemType m_default_activity;

public:
    DeclareConstructorFromConfigWithNumInputs(SequenceDecoderNode);
    SequenceDecoderNode(DEVICEID_TYPE deviceId, const wstring& name)
        : Base(deviceId, name),
          mAlpha(deviceId),
          mBacktrace(deviceId),
          mStartLab(-1),
          mEndLab(-1)
    {
    }

    static void DecideStartEndingOutputLab(const Matrix<ElemType>& lbls, int& stt, int& stp)
    {
        if (stt != -1 && stp != -1)
            return; // have computed before

        int iNumPos = lbls.GetNumCols();

        int firstLbl = -1;
        for (int ik = 0; ik < lbls.GetNumRows(); ik++)
            if (lbls(ik, 0) != 0)
            {
                firstLbl = ik;
                break;
            }

        int lastLbl = -1;
        for (int ik = 0; ik < lbls.GetNumRows(); ik++)
            if (lbls(ik, iNumPos - 1) != 0)
            {
                lastLbl = ik;
                break;
            }

        stt = firstLbl;
        stp = lastLbl;
    };

    virtual void BackpropToNonLooping(size_t /*inputIndex*/) override // scaled by 2*number of elements in the Matrix<ElemType>
    {
        LogicError("SequenceDecoder is used for evaluation only.");
    }

    virtual bool OutputUsedInComputingInputNodesGradients() const override
    {
        return false;
    }
    virtual bool InputUsedInComputingInputNodesGradients(size_t /*childIndex*/) const override
    {
        return false;
    }

    // compute posterior probability of label y at position t
    virtual void /*ComputationNodeNonLooping::*/ ForwardPropNonLooping() override
    {
        DecideStartEndingOutputLab(InputRef(0).Value(), mStartLab, mEndLab);
        ForwardPropS(mAlpha, mBacktrace, Value(), InputRef(1).Value(),
                     InputRef(2).Value(), mStartLab, mEndLab);
    }

    // compute forward backward algorithm
    void ForwardPropS(Matrix<ElemType>& alpha, Matrix<ElemType>& backtrace, Matrix<ElemType>& functionValues, const Matrix<ElemType>& pos_scores, const Matrix<ElemType>& pair_scores, const size_t stt, const size_t stp)
    {
        // to-do, each slice is for one sentence
        // to-do, number of slices correspond to number of frames
        // this implementation only supports one sentence per minibatch

        // change to other values so can support multiple sentences in each minibatch
        ForwardCompute(alpha, backtrace, pos_scores, pair_scores, stt);
        BackwardCompute(functionValues, backtrace, stp);
    };

    // compute forward backward algorithm
    static void ForwardCompute(Matrix<ElemType>& alpha,
                               Matrix<ElemType>& backtrace,
                               const Matrix<ElemType>& pos_scores, const Matrix<ElemType>& pair_scores,
                               const size_t stt)
    {
        // to-do, shift more than 1 to support muliple sentences per minibatch
        int iNumPos = pos_scores.GetNumCols();
        int iNumLab = pos_scores.GetNumRows();
        size_t iTmp = 0;

        // need to have
        alpha.Resize(iNumLab, iNumPos);
        backtrace.Resize(iNumLab, iNumPos);

        for (int t = 0; t < iNumPos; t++)
        {
            for (int k = 0; k < iNumLab; k++)
            {
                ElemType fTmp = (ElemType) LZERO;
                if (t > 1)
                {
                    for (int j = 0; j < iNumLab; j++)
                    {
                        ElemType fAlpha = alpha(j, t - 1) + pair_scores(k, j);
                        if (fAlpha > fTmp)
                        {
                            fTmp = fAlpha;
                            iTmp = j;
                        }
                    }
                    fTmp += pos_scores(k, t); // include position dependent score
                }
                else
                {
                    // with constrain that the first word is labeled as a given symbol
                    iTmp = stt;
                    fTmp = 0;
                    if (t == 1)
                    {
                        fTmp = alpha(iTmp, t - 1);
                        fTmp += pair_scores(k, iTmp);
                        fTmp += pos_scores(k, t);
                    }
                    else
                    {
                        fTmp = (k == stt) ? pos_scores(k, t) : (ElemType) LZERO;
                    }
                }
                alpha(k, t) = fTmp;
                backtrace(k, t) = (ElemType) iTmp;
            }
        }
    };

    // compute backward algorithm
    static void BackwardCompute(
        Matrix<ElemType>& decodedpath,
        const Matrix<ElemType>& backtrace, const size_t stp)
    {
        int iNumPos = backtrace.GetNumCols();
        int iNumLab = backtrace.GetNumRows();

        decodedpath.Resize(iNumLab, iNumPos);
        decodedpath.SetValue(0);

        size_t lastlbl = stp;
        decodedpath(lastlbl, iNumPos - 1) = 1;

        for (int t = iNumPos - 1; t > 0; t--)
        {
            lastlbl = (size_t) backtrace(lastlbl, t);
            decodedpath(lastlbl, t - 1) = 1;
        }
    };

    // need to feed in pseudo label data, which tells the decoder what is the beginning
    // and ending output symbol. these symbols will constrain the search space
    virtual void /*ComputationNodeBase::*/ Validate(bool isFinalValidationPass) override
    {
        Base::Validate(isFinalValidationPass);
        InferMBLayoutFromInputsForStandardCase(isFinalValidationPass);

        if (isFinalValidationPass)
            if (!(Input(1)->GetSampleMatrixNumRows() == Input(2)->GetSampleMatrixNumRows() && // position dependent and pair scores have same number of labels
                  Input(0)->GetSampleMatrixNumRows() == Input(1)->GetSampleMatrixNumRows() &&
                  Input(0)->GetSampleMatrixNumCols() == Input(1)->GetSampleMatrixNumCols() && // position dependent and pair scores have the same observation numbers
                  Input(2)->GetSampleMatrixNumCols() == Input(2)->GetSampleMatrixNumRows()))
            {
                LogicError("The Matrix<ElemType>  dimension in the SequenceDecoderNode operation does not match.");
            }
        // BUGBUG: No SetDims()?
        m_sampleLayout = TensorShape();
    }
};

template class SequenceDecoderNode<float>;
template class SequenceDecoderNode<double>;

#endif

} } }
