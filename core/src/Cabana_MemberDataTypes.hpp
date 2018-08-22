#ifndef CABANA_MEMBERDATATYPES_HPP
#define CABANA_MEMBERDATATYPES_HPP

#include <cstdlib>
#include <type_traits>

namespace Cabana
{
//---------------------------------------------------------------------------//
/*!
 \class MemberDataTypes
 \brief General sequence of types for SoA and AoSoA member data.
*/
template<typename... Types>
struct MemberDataTypes
{
    static constexpr std::size_t size = sizeof...(Types);
};

//---------------------------------------------------------------------------//
/*!
  \class MemberTag
  \brief Tag for member data type indices.
*/
template<std::size_t I>
struct MemberTag : public std::integral_constant<std::size_t,I> {};

//---------------------------------------------------------------------------//
/*!
  \class MemberDataTypeAtIndex
  \brief Get the type of the member at a given index.
*/
template<std::size_t I, typename T, typename... Types>
struct MemberDataTypeAtIndexImpl;

template<typename T, typename... Types>
struct MemberDataTypeAtIndexImpl<0,T,Types...>
{
    using type = T;
};

template<std::size_t I, typename T, typename... Types>
struct MemberDataTypeAtIndexImpl
{
    using type = typename MemberDataTypeAtIndexImpl<I-1,Types...>::type;
};

template<std::size_t I, typename... Types>
struct MemberDataTypeAtIndex;

template<std::size_t I, typename... Types>
struct MemberDataTypeAtIndex<I,MemberDataTypes<Types...> >
{
    using type =
        typename MemberDataTypeAtIndexImpl<I,Types...>::type;
};

//---------------------------------------------------------------------------//

} // end namespace Cabana

#endif // end CABANA_MEMBERDATATYPES_HPP
