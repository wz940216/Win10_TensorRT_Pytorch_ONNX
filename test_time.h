class CSpendTime
{
public:
	CSpendTime()
	{
		m_dSpendTime = 0.0;

		QueryPerformanceFrequency(&m_lgiFrequency);
	}

	virtual ~CSpendTime()
	{
	}

	// 开始
	void Start()
	{
		QueryPerformanceCounter(&m_lgiCounterStart);
	}

	// 结束
	double End() // 返回值：耗时（单位：毫秒）
	{
		QueryPerformanceCounter(&m_lgiCounterEnd);

		m_dSpendTime = (double)(m_lgiCounterEnd.QuadPart - m_lgiCounterStart.QuadPart) * 1000.0 / m_lgiFrequency.QuadPart;

		return m_dSpendTime;
	}

	//CString EndS()
	//{
	//	double dTime = End();
	//	CString strTime;
	//	strTime.Format(_T("  %.3f ms "), dTime);
	//	return strTime;
	//}


	// 获得耗时（单位：毫秒）
	int GetMillisecondInt()
	{
		return (int)(m_dSpendTime);
	}

	// 获得耗时（单位：毫秒）
	double GetMillisecondDouble()
	{
		return (m_dSpendTime);
	}

protected:

	LARGE_INTEGER m_lgiCounterStart;
	LARGE_INTEGER m_lgiCounterEnd;
	LARGE_INTEGER m_lgiFrequency;
	double m_dSpendTime;
};